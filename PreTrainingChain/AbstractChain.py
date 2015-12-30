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

from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainer import ChainList, optimizers
import numpy as np
import six
import pt_linear as P


class AbstractChain(ChainList):
    """
    [Classes]
    Extension of chainer.ChainList.
    feature:
    1) You can define network structure by list or tuple such as [784, 250, 200, 160, 10].
       This feature accelerate your Deep network development.
       If you call this class by AbstractChain([784, 250, 200, 160, 10]),
       ChainList->
       F.Linear(784, 250)
       F.Linear(250, 200)
       F.Linear(200, 160)
       F.Linear(160, 10)
       You can change total layers without any hard coding.

    2) Pre-training is implemented.
       You can use it only by calling AbstractChain.pre_training(train_data, test_data)
       test_data is optional.
       If you input any test_data, result of test as autoencoder at each hidden layer will be printed.
       If length of train_Data is zero, Pre-training is skipped.

    3) This class is super class of ChainList.
       So you can use function of ChainList.
    """
    def __init__(self, n_units, epoch=10, batch_size=100, visualize=True):
        ChainList.__init__(self)
        self.n_units = n_units[0:-1]
        self.last_unit = n_units[-1]
        self.total_layer = len(n_units)
        self.collect_child_model()
        self.set_optimizer()
        self.epoch = epoch
        self.batch_size = batch_size
        self.visualize = visualize
        self.pre_trained = False

    def set_optimizer(self):
        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self)

    def collect_child_model(self):
        self.child_models = []
        for i, n_unit in enumerate(self.n_units):
            if i == 0: continue
            self.child_models.append(ChildChainList(P.PTLinear(self.n_units[i-1], n_unit)))
            #self.child_models.append(ChildChainList(F.Linear(self.n_units[i-1], n_unit)))

    def forward(self, x_data, train=True):
        data = x_data
        for model in self:
            data = F.dropout(F.relu(model(data)), train=train)
        return data

    def pre_training(self, sample, test=[]):
        """
        [FUNCTIONS]
        Do Pre-training for each layers by using Auto-Encoder method.
        """
        P.PT_manager(F.relu)
        now_sample = sample
        now_test = test
        if sample.size:
            for child in self.child_models:
                P.PT_manager().is_pre_training = True
                child.learn_as_autoencoder(now_sample, now_test)
                P.PT_manager().is_pre_training = False
                self.add_link(child[0].copy())
                now_sample = self.forward(Variable(sample), False).data
                if len(test):
                    now_test = self.forward(Variable(test), False).data
        else:
            for child in self.child_models:
                self.add_link(child[0].copy())
                #self.add_link(child[0])
        self.add_last_layer()
        self.pre_trained = True

    def add_last_layer(self):
        raise NotImplementedError("""`add_last_layer` method is not implemented.
        You have to link self.n_units[-1] and self.last_unit
        example)
        self.add_link(F.Linear(self.n_units[-1], self.last_unit))""")

    def loss_function(self, x, y):
        raise NotImplementedError("""`loss_function` method is not implemented.
        example)
        return F.softmax_cross_entropy(x, y)""")

    def learn(self, x_train, y_train, x_test, y_test, isClassification=False):
        if not self.pre_trained:
            self.pre_training(np.array([]), np.array([]))
        train_size = x_train.shape[0]
        train_data_size = x_train.shape[1]

        for epoch in six.moves.range(self.epoch):
            perm = np.random.permutation(train_size)
            train_loss = 0.
            test_loss = 0.
            test_accuracy = 0.
            for i in range(0, train_size, self.batch_size):
                x = Variable(x_train[perm[i:i+self.batch_size]])
                y = Variable(y_train[perm[i:i+self.batch_size]])
                self.zerograds()
                loss = self.loss_function(self.forward(x, train=True), y)
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            if len(x_test):
                x = Variable(x_test)
                y = Variable(y_test)
                predict = self.forward(x, train=False)
                test_loss = self.loss_function(predict, y).data
                print('test_loss: ' + str(test_loss))
                if isClassification:
                    test_accuracy = F.accuracy(predict, y).data
                    print('test_accuracy: ' + str(test_accuracy))

        if self.visualize:
            self.visualize_net(loss)

    def visualize_net(self, loss):
        import chainer.computational_graph as c
        g = c.build_computational_graph((loss,))
        with open('graph.dot', 'w') as o:
            o.write(g.dump())


class ChildChainList(ChainList):
    """
    [Classes]
    This class mustn't be called directoly.
    Have to be called by super class of AbstractChain.
    This chain will be learn as autoencoder,
    so I don't expect to configurate forward, forward_as_autoencoder and learn_as_autoencoder.
    But you can configurate optimizer by editting __init__.
    """
    def __init__(self, link, epoch=10, batch_size=100, visualize=True):
        ChainList.__init__(self, link)
        self.optimizer = optimizers.AdaDelta()
        self.optimizer.setup(self)
        self.loss_function = F.mean_squared_error
        self.epoch = epoch
        self.batch_size = batch_size
        self.visualize = visualize

    def forward(self, x_data, train):
        return F.dropout(F.relu(self[0](x_data)), train=train)

    def learn_as_autoencoder(self, x_train, x_test=None):
        optimizer = self.optimizer
        train_size = x_train.shape[0]
        train_data_size = x_train.shape[1]
        #self.add_dummy_output_link(train_data_size)
        for epoch in six.moves.range(self.epoch):
            perm = np.random.permutation(train_size)
            train_loss = 0
            test_loss = None
            test_accuracy = 0
            for i in range(0, train_size, self.batch_size):
                x = Variable(x_train[perm[i:i+self.batch_size]])
                self.zerograds()
                loss = self.loss_function(self[0](x), x)
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            if len(x_test):
                x = Variable(x_test)
                test_loss = self.loss_function(self[0](x), x).data
        if test_loss is not None:
            print('Pre-training test loss: ' + str(test_loss))
        if self.visualize:
            import chainer.computational_graph as c
            g = c.build_computational_graph((loss,))
            with open('child_graph.dot', 'w') as o:
                o.write(g.dump())
        del self.optimizer
