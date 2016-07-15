#-------------------------------------------------------------------------------
# Name:        PretrainingChain
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     Apache Licence 2.0
#-------------------------------------------------------------------------------

from chainer import Variable
from chainer import optimizers
from AbstractChain import AbstractChain
import chainer.functions as F
import chainer.links as L
import numpy as np
import six


class MaxoutClassifier(AbstractChain):
    isClassification = True
    def __init__(self, n_units, pool_size, epoch=10, batch_size=100,
                 dropout_rate=(), optimizer=optimizers.AdaDelta()):
        super(MaxoutClassifier, self).__init__(n_units, epoch, batch_size,
                                              dropout_rate, optimizer)
        self.pool_size = pool_size

    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

    def add_last_layer(self):
        self.add_link(L.Maxout(self.n_units[-1], self.last_unit, self.pool_size))

    def predict_proba(self, x):
        if not self.fit__:
            raise Exception('Call predict before fit.')
        return self.forward(Variable(x), False).data

    def predict(self, x):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def __collect_child_model(self):
        self.child_models = []
        for i, n_unit in enumerate(self.n_units):
            if i == 0: continue
            self.child_models.append(ChildChainList(L.Maxout(self.n_units[i-1], n_unit, self.pool_size)))
