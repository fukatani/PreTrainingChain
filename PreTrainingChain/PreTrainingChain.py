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
from AbstractChain import AbstractChain
import chainer.functions as F
import numpy as np
import six


class ChainClassfier(AbstractChain):
    isClassification = True
    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit))

    def predict_proba(self, x):
        if not self.fit__:
            raise Exception('Call predict before fit.')
        return self.forward(Variable(x), False).data

    def predict(self, x):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class ChainRegression(AbstractChain):
    isClassification = False
    def loss_function(self, x, y):
        return F.MeanSquaredError(x, y)

    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit))
