 #!/usr/bin/env python
import random
import argparse
import numpy
import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class SmallClassificationModel(Chain):
    def __init__(self):
        super(SmallClassificationModel, self).__init__(
            fc1 = L.Linear(2, 2)
            )
    def _forward(self, x):
        h = self.fc1(x)
        return h

    def train(self, x_data, y_data):
        x = Variable(x_data.reshape(1,2).astype(numpy.float32))
        y = Variable(y_data.astype(numpy.int32))
        h = self._forward(x)

        L.zerograds()
        error = F.softmax_cross_entropy(h, y)
        accuracy = F.accuracy(h, y)
        error.backward()
        optimizer.update()
        print("x: {}".format(x.data))
        print("h: {}".format(h.data))
        print("h_class: {}".format(h.data.argmax()))
        #print("error: {}".format(error.data[0]))
        #print("accuracy: {}".format(accuracy.data))

class ClassificationModel(Chain):
    def __init__(self):
        super(ClassificationModel, self).__init__(
            fc1 = L.Linear(2, 2),
            fc2 = L.Linear(2, 2)
            )
    def _forward(self, x):
        h = self.fc2(F.sigmoid(self.fc1(x)))
        return h

    def train(self, x_data, y_data):
        x = Variable(x_data.reshape(1,2).astype(numpy.float32), )
        y = Variable(y_data.astype(numpy.int32))
        h = self._forward(x)
        optimizer.zerograds()
        error = F.softmax_cross_entropy(h, y)
        accuracy = F.accuracy(h, y)
        error.backward()
        optimizer.update()
        print("x: {}".format(x.data))
        print("h: {}".format(h.data))
        print("h_class: {}".format(h.data.argmax()))

class RegressionModel(Chain):
    def __init__(self):
        super(RegressionModel, self).__init__(
            fc1 = L.Linear(2, 2),
            fc2 = L.Linear(2, 1)
            )

    def __call__(self, x, y):
        fv = self.fwd(x)
        loss = F.mean_squared_error(fv, y)
        return loss

    def fwd(self, x):
        return F.sigmoid(self.fc1(x))
"""
    def train(self, x_data, y_data):
        x = Variable(x_data.reshape(1,2).astype(numpy.float32))
        y = Variable(y_data.astype(numpy.float32))
        self.zerograds()
        loss.backward()
        optimizer.update()
        print("x: {}".format(x.data))
        print("h: {}".format(h.data))
"""

model = RegressionModel()
#model = ClassificationModel()
#model = ClassificationModel()
#optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer = optimizers.SGD()
optimizer.setup(model)

data_xor = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([1])],
    [numpy.array([1,0]), numpy.array([1])],
    [numpy.array([1,1]), numpy.array([0])],
]*1000

data_and = [
    [numpy.array([0,0]), numpy.array([0])],
    [numpy.array([0,1]), numpy.array([0])],
    [numpy.array([1,0]), numpy.array([0])],
    [numpy.array([1,1]), numpy.array([1])],
]*1000

for invec, outvec in data_xor:
    invec = Variable(invec.reshape(1,2).astype(numpy.float32))
    outvec = Variable(outvec.astype(numpy.float32))
    model.zerograds()
    loss = model(invec, outvec)
    loss.backward()
    optimizer.update()
    print("x: {}".format(x.data))
    print("h: {}".format(fv.data))
