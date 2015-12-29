import math

from chainer import cuda
from chainer import function
from chainer.functions import Sigmoid
from chainer.utils import type_check

import numpy

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class Pretrainable_Linear(function.Function):

    def __init__(self, in_size, hidden_size, activation=Sigmoid,
                 wscale=1, bias=0,
                 initialW=None, initial_bias1=None, initial_bias2=None):
        self.W = None
        self.gW = None
        self.b1 = None
        self.b2 = None
        self.gb1 = None
        self.gb2 = None
        self.activation = None
        self.name = None

        if initialW is not None:
            assert initialW.shape == (hidden_size, in_size)
            self.W = initialW
        else:
            self.W = numpy.random.normal(
                0, wscale * math.sqrt(1. / in_size),
                (hidden_size, in_size)).astype(numpy.float32)
        xp = cuda.get_array_module(self.W)
        self.gW = xp.full_like(self.W, numpy.nan)

        if initial_bias1 is not None:
            assert initial_bias1.shape == (hidden_size,)
            self.b1 = initial_bias1
        else:
            self.b1 = numpy.repeat(numpy.float32(bias), hidden_size)

        if initial_bias2 is not None:
            assert initial_bias2.shape == (in_size,)
            self.b2 = initial_bias2
        else:
            self.b2 = numpy.repeat(numpy.float32(bias), in_size)

        self.gb1 = xp.empty_like(self.b1)
        self.gb2 = xp.empty_like(self.b2)

        if activation is not None:
            if activation == Sigmoid:
                self.activation = activation()
            else:
                self.activation = activation
        self.is_pre_training = True

    def hidden(self, x):
        h = _Encoder(self.W, self.b1)(x)
        if self.activation is not None:
            h = self.activation(h)
        h.unchain_backward()
        return h

    @property
    def parameter_names(self):
        return 'W', 'b1', 'b2'

    @property
    def gradient_names(self):
        return 'gW', 'gb1', 'gb2'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            (type_check.Variable(numpy.prod, 'prod')(x_type.shape[1:]) ==
             type_check.Variable(self.W.shape[1], 'W.shape[1]')),
        )

    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() == 1,
            out_types.size() == 1,
        )
        x_type, = in_types
        y_type, = out_types

        type_check.expect(
            y_type.dtype == numpy.float32,
            y_type.ndim == 2,
            y_type.shape[0] == x_type.shape[0],
            y_type.shape[1] == type_check.Variable(self.W.shape[1],
                                                   'W.shape[1]'),
        )

    def zero_grads(self):
        self.gW.fill(0)
        self.gb1.fill(0)
        self.gb2.fill(0)

    def forward(self, x):
        _x = _as_mat(x[0])
        Wx = _x.dot(self.W.T)
        Wx += self.b1

        self.x_activation = Wx
        if self.activation is not None:
            h, = self.activation.forward([Wx])
        else:
            h = Wx
        self.x_decode = h
        if not self.is_pre_training:
            return h
        y = h.dot(self.W)
        y += self.b2

        return y,

    def backward(self, x, gy):
        _x = self.x_decode
        _gy = gy[0]
        self.gW += _x.T.dot(_gy)
        self.gb2 += _gy.sum(0)
        _gy = _gy.dot(self.W.T).reshape(_x.shape)

        if self.activation is not None:
            _gy, = self.activation.backward([self.x_activation], [_gy])

        _x = _as_mat(x[0])
        self.gW += _gy.T.dot(_x)
        self.gb1 += _gy.sum(0)

        return _gy.dot(self.W).reshape(x[0].shape),

# undifferentiable Linear function
class _Encoder(function.Function):

    def __init__(self, initialW, initial_Bias):
        self.W = initialW
        self.b = initial_Bias

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            (type_check.Variable(numpy.prod, 'prod')(x_type.shape[1:]) ==
             type_check.Variable(self.W.shape[1], 'W.shape[1]')),
        )

    def forward(self, x):
        x = _as_mat(x[0])
        Wx = x.dot(self.W.T)
        Wx += self.b
        return Wx,
