#-------------------------------------------------------------------------------
# Name:        util
# Purpose:
#
# Author:      rf
#
# Created:     04/24/2016
# Copyright:   (c) rf 2016
# Licence:     Apache Licence 2.0
#-------------------------------------------------------------------------------

import numpy as np

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

