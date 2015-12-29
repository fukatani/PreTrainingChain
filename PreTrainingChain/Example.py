#-------------------------------------------------------------------------------
# Name:        Example
# Purpose:
#
# Author:      rf
#
# Created:     11/08/2015
# Copyright:   (c) rf 2015
# Licence:     Apache Licence 2.0
#-------------------------------------------------------------------------------

import numpy as np
import PreTrainingChain.AbstractChain

class PreTrainingDNN(PreTrainingChain.AbstractChain.AbstractChain):
    """
    [Classes]
    Sample of DNN for classification.
    If you use this class for minst.
    n_units = [784, n, m, ..., 10]
    784 is dimension of sample.data and 10 is dimension of sample.target.
    """
    def add_last_layer(self):
        self.add_link(F.Linear(self.n_units[-1], self.last_unit))

    def loss_function(self, x, y):
        return F.softmax_cross_entropy(x, y)

def make_sample(size):
    from sklearn.datasets import fetch_mldata
    print('fetch MNIST dataset')
    sample = fetch_mldata('MNIST original')
    perm = np.random.permutation(len(sample.data))
    sample.data = sample.data[perm[0: size]]
    sample.target = sample.target[perm[0: size]]
    print('Successed data fetching')
    sample.data   = sample.data.astype(np.float32)
    sample.data  /= 255
    sample.target = sample.target.astype(np.int32)
    return sample

if __name__ == '__main__':
    pre_train_size = 1000
    pre_test_size = 200
    train_size = 1000
    test_size = 200

    sample = make_sample(pre_train_size+pre_test_size+train_size+test_size)
    x_pre_train, x_pre_test, x_train, x_test, _ = np.split(sample.data,
        [pre_train_size,
        pre_train_size + pre_test_size,
        pre_train_size + pre_test_size + train_size,
        pre_train_size + pre_test_size + train_size + test_size])

    _, _, y_train, y_test, _ = np.split(sample.target,
        [pre_train_size,
        pre_train_size + pre_test_size,
        pre_train_size + pre_test_size + train_size,
        pre_train_size + pre_test_size + train_size + test_size])

    pc = PreTrainingDNN([784,200,200,200,200,10])
    pc.pre_training(x_pre_train, x_pre_test)
    pc.learn(x_train, y_train, x_test, y_test, True)

