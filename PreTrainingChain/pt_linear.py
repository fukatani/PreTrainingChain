import numpy
from chainer import cuda
from chainer import function
from chainer import flag
from chainer.utils import type_check
from chainer import variable
import chainer.functions as F
import weakref
from chainer import link
from chainer.functions import relu


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class PT_manager(object):
    """ [CLASSES]
        Singleton class for manage terminals for DFxxx.
    """
    _singleton = None
    def __new__(cls, activation=None):
        if cls._singleton == None:
            cls._singleton = object.__new__(cls)
            cls.is_pre_training = True
            assert activation is not None
        return cls._singleton

    def get_pt_info(self):
        return self.is_pre_training, F.ReLU#self.activation

class PTLinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in >=2, n_in <= 4)
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

    def forward(self, inputs):
        is_pre_training, activation = PT_manager().get_pt_info()
        x = _as_mat(inputs[0])
        W = inputs[1]
        h = x.dot(W.T)
        if len(inputs) == 4:
            b = inputs[2]
            h += b
        if not is_pre_training:
            return h, #not using activation

        self.x_activation = h
        h, = activation().forward((h,))
        self.x_decode = h
        y = h.dot(W)
        if len(inputs) == 4:
            b2 = inputs[3]
            y += b2
        return y,

    def backward(self, inputs, grad_outputs):
        is_pre_training, activation = PT_manager().get_pt_info()
        if not is_pre_training:
            x = _as_mat(inputs[0])
            W = inputs[1]
            gy = grad_outputs[0]

            gx = gy.dot(W).reshape(inputs[0].shape)
            gW = gy.T.dot(x)
            if len(inputs) >= 3:
                gb = gy.sum(0)
                return gx, gW, gb, numpy.zeros(x.shape[1]).astype(numpy.float32) #gb2=0
            else:
                return gx, gW
        else:
            x = self.x_decode
            W = inputs[1]
            gy = grad_outputs[0]
            gW = x.T.dot(gy)
            if len(inputs) == 4:
                gb2 = gy.sum(0)
            gy = gy.dot(W.T).reshape(x.shape)
            gy, = activation().backward([self.x_activation], [gy])
            x = _as_mat(inputs[0])
            gW += gy.T.dot(x)

            #gx = gy.dot(W).reshape(x[0].shape),
            gx = gy.dot(W).reshape(x.shape)
            if len(inputs) == 4:
                gb = gy.sum(0)
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
        return PTLinearFunction()(x, W)
    else:
        return PTLinearFunction()(x, W, b, b2)

class PTLinear(link.Link):

    """Pre trainable Linear layer (a.k.a. fully-connected layer).

    This is a link that wraps the :func:`~chainer.functions.pt_linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.

    The weight matrix ``W`` is initialized with i.i.d. Gaussian samples, each
    of which has zero mean and deviation :math:`\sqrt{1/\\text{in_size}}`. The
    bias vector ``b`` is of size ``out_size``. Each element is initialized with
    the ``bias`` value. If ``nobias`` argument is set to True, then this link
    does not hold a bias vector.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """
    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None, initial_bias2=None):
        super(PTLinear, self).__init__(W=(out_size, in_size))
        if initialW is None:
            initialW = numpy.random.normal(
                0, wscale * numpy.sqrt(1. / in_size), (out_size, in_size))
        self.W.data[...] = initialW

        if nobias:
            self.b = None
            self.b2 = None
        else:
            self.add_param('b', out_size)
            self.add_param('b2', in_size)
            if initial_bias is None:
                initial_bias = bias
                initial_bias2 = bias
            self.b.data[...] = initial_bias
            self.b2.data[...] = initial_bias2

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        return ptlinear(x, self.W, self.b, self.b2)