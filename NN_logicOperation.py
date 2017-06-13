#!/usr/bin/env python

# 参考元 : http://hi-king.hatenablog.com/entry/2015/06/27/194630

import random
import argparse
import numpy
import chainer
import chainer.optimizers


class SmallClassificationModel(chainer.Chain):
    def __init__(self):
        super(SmallClassificationModel, self).__init__(
            l1 = chainer.links.Linear(2, 2)
            )
    def _forward(self, x):
        h = self.l1(x)
        return h

    def train(self, x_data, y_data):
        x = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32))
        y = chainer.Variable(y_data.astype(numpy.int32))
        h = self._forward(x)

        self.zerograds()
        error = chainer.functions.softmax_cross_entropy(h, y)
        accuracy = chainer.functions.accuracy(h, y)
        error.backward()
        optimizer.update()
        if epoch%100==0:
            print(' %d & %d = %d (zero:%f one:%f)' % (x.data[0,0], x.data[0,1], h.data.argmax(), h.data[0,0], h.data[0,1]))
        #print("error: {}".format(error.data[0]))
        #print("accuracy: {}".format(accuracy.data))

class ClassificationModel(chainer.Chain):
    def __init__(self):
        super(ClassificationModel, self).__init__(
            l1 = chainer.links.Linear(2, 2),
            l2 = chainer.links.Linear(2, 2)
            )
    def _forward(self, x):
        h = self.l2(chainer.functions.sigmoid(self.l1(x)))
        return h

    def train(self, x_data, y_data, epoch):
        x = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32))
        y = chainer.Variable(y_data.astype(numpy.int32))
        h = self._forward(x)

        self.zerograds()
        error = chainer.functions.softmax_cross_entropy(h, y)
        accuracy = chainer.functions.accuracy(h, y)
        error.backward()
        optimizer.update()
        if epoch%100==0:
            print(' %d & %d = %d (zero:%f one:%f)' % (x.data[0,0], x.data[0,1], h.data.argmax(), h.data[0,0], h.data[0,1]))

class RegressionModel(chainer.Chain):
    def __init__(self):
        super(RegressionModel, self).__init__(
            l1 = chainer.links.Linear(2, 2),
            l2 = chainer.links.Linear(2, 1)
            )

    def _forward(self, x):
        h = self.l2(chainer.functions.sigmoid(self.l1(x)))
        return h

    def train(self, x_data, y_data, epoch):
        x = chainer.Variable(x_data.reshape(1,2).astype(numpy.float32))
        y = chainer.Variable(y_data.reshape(1,1).astype(numpy.float32))
        h = self._forward(x)
        self.zerograds()
        error = chainer.functions.mean_squared_error(h, y)
        error.backward()
        optimizer.update()
        if epoch%100==0:
                print('x: {}  h: {})'.format(x.data, h.data))


#model = SmallClassificationModel()     # 層が1つの場合
model = ClassificationModel()           # 層が2つの場合
#model = RegressionModel()              # 重回帰でやった場合

optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

data_xor = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([1])],
    [numpy.array([1,0]), numpy.array([1])],
    [numpy.array([1,1]), numpy.array([0])],
]

data_and = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([0])],
    [numpy.array([1,0]), numpy.array([0])],
    [numpy.array([1,1]), numpy.array([1])],
]

data_or = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([1])],
    [numpy.array([1,0]), numpy.array([1])],
    [numpy.array([1,1]), numpy.array([1])],
]

for epoch in range(1001):
    if epoch%100==0:
        print("epoch: %d" %epoch)
    for invec, outvec in data_xor:
        model.train(invec, outvec, epoch)
