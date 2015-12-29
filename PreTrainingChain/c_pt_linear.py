import numpy
from chainer import cuda
from chainer import function
from chainer import flag
from chainer.utils import type_check
from chainer import variable
import weakref


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)
#TODO
class StateFunction(function.Function):

    def __call__(self, is_pre_training, activation,*inputs):
        in_data = tuple([x.data for x in inputs])
        if self.type_check_enable:
            self._check_data_type_forward(in_data)
        # Forward prop
        with cuda.get_device(*in_data):
            outputs = self.forward(in_data, is_pre_training, activation)
            assert type(outputs) == tuple

        out_v = flag.aggregate_flags([x.volatile for x in inputs])
        ret = tuple([variable.Variable(y, volatile=out_v) for y in outputs])

        if out_v != 'on':
            # Topological ordering
            self.rank = max([x.rank for x in inputs]) if inputs else 0
            # Backward edges
            for y in ret:
                y.set_creator(self)
            self.inputs = inputs
            # Forward edges (must be weak references)
            self.outputs = tuple([weakref.ref(y) for y in ret])

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _check_data_type_forward(self, in_data):
        in_type = type_check.get_types(in_data, 'in_types', False)
        try:
            self.check_type_forward(in_type)
        except type_check.InvalidType as e:
            msg = """
Invalid operation is performed in: {0} (Forward)

{1}""".format(self.label, str(e))
            raise type_check.InvalidType(e.expect, e.actual, msg=msg)

class PTLinearFunction(StateFunction):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == numpy.float32,
            w_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == numpy.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs, is_pre_training, activation):
        x = _as_mat(inputs[0])
        W = inputs[1]
        h = x.dot(W.T)
        if len(inputs) == 4:
            b = inputs[2]
            h += b
        if not is_pre_training:
            return h, #not using activation

        self.x_activation = h
        h = activation(variable.Variable(h)).data
        self.x_decode = h
        y = h.dot(W)
        if len(inputs) == 4:
            b2 = inputs[3]# TODO
            y += b2
        return y,

    def backward(self, inputs, grad_outputs, is_pre_training, activation):
        if not is_pre_training:
            x = _as_mat(inputs[0])
            W = inputs[1]
            gy = grad_outputs[0]

            gx = gy.dot(W).reshape(inputs[0].shape)
            gW = gy.T.dot(x)
            if len(inputs) == 3:
                gb = gy.sum(0)
                return gx, gW, gb
            else:
                return gx, gW
        else:
            x = self.x_decode
            gy = grad_outputs[0]
            gW += x.T.dot(gy)
            if len(inputs) == 4:
                gb2 += gy.sum(0)
            gy = gy.dot(W.T).reshape(x.shape)
            gy, = activation.backward([self.x_activation], [gy])
            x = _as_mat(inputs[0])
            gW += _gy.T.dot(x)

            gx = _gy.dot(self.W).reshape(x[0].shape),
            if len(inputs) == 4:
                gb += gy.sum(0)
                return gx, gW, gb, gb2
            else:
                return gx, gW

def ptlinear(x, W, b=None, b2=None, is_pre_training=False, activation=None):
    """Pre trainable Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``..
        is_pre_training: boolean
        activation: function for pre-training auto encoder.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """
    if b is None:
        return PTLinearFunction()(is_pre_training, activation, x, W)
    else:
        return PTLinearFunction()(is_pre_training, activation, x, W, b)


