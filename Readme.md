
##Introduction


Extension of chainer.
ChainList for the purpose of network scalability/congirablity/Pre-training executablity for deep leaning.
(You need to get deep learning framework "chainer" from http://chainer.org/)

## feature:
####1) You can define network structure by list or tuple such as [784, 250, 200, 160, 10].
   This feature accelerate your deep network development.
   If you call this class by ChainClassfier([784, 250, 200, 160, 10]),
   you can generate ChainList->
   (F.Linear(784, 250),
   F.Linear(250, 200),
   F.Linear(200, 160),
   F.Linear(160, 10))
   You can change network structure without any hard coding.

####2) Pre-training executable.
   You can execute pre-training only by calling AbstractChain.pre_training(train_data).
   Pretraining is executed by using Bengio method.
   (http://arxiv.org/pdf/1206.5538.pdf)
   If length of train_Data is zero, Pre-training is skipped.
   
####3)Usage as scikit-learn library, and correpond to GridSearch parameter tuning.
You can use PreTraining_chain as scikit-learn library, So ChainClassfier.fit, ChainClassfier.predict, ChainClassfier.score is usable.
Also you can use sklearn.gridsearchCV.
Please see [GridSearchExample.py.](https://github.com/fukatani/PreTrainingChain/blob/master/PreTrainingChain/GridSearchExample.py)


##Software Requirements

* Python (2.7)
* chainer >= 1.8.0
* scikit-learn

##Installation


```
$ pip install PreTrainingChain
```

or

```
$ git clone https://github.com/fukatani/PreTrainingChain.git
```

##Example

Implement example is here
https://github.com/fukatani/PreTrainingChain/blob/master/PreTrainingChain/Example.py
You have to override add_last_layer method and loss_function method.

Example.py is implement for  mnist classification.

```
$ python Example.py

fetch MNIST dataset
Successed data fetching
Pre-training test loss: 0.0895392745733
Pre-training test loss: 0.000182752759429
Pre-training test loss: 5.92054857407e-05
Pre-training test loss: 1.82532239705e-05
test_loss: 2.30244994164
test_accuracy: 0.0799999982119
test_loss: 2.30086517334
test_accuracy: 0.189999997616
test_loss: 2.28533029556
test_accuracy: 0.27500000596
test_loss: 2.25788879395
test_accuracy: 0.294999986887
test_loss: 2.21044063568
test_accuracy: 0.284999996424
test_loss: 2.13255786896
test_accuracy: 0.280000001192
test_loss: 2.09592270851
test_accuracy: 0.305000007153
test_loss: 2.05419230461
test_accuracy: 0.294999986887
test_loss: 2.04007315636
test_accuracy: 0.294999986887
test_loss: 2.01762104034
test_accuracy: 0.289999991655
```


##License

Apache License 2.0
(http://www.apache.org/licenses/LICENSE-2.0)


##Copyright

Copyright (C) 2015, Ryosuke Fukatani

##Related Project and Site

chainer
http://docs.chainer.org/en/stable/index.html

Blog entry(Japanese)
http://segafreder.hatenablog.com/entry/2015/12/30/183319
