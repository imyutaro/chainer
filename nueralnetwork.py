import numpy as np
import chainer
from chainer import Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
from chainer.functions.loss.mean_squared_error import mean_squared_error



x1 = Variable(np.array([1], dtype=np.float32))
x2 = Variable(np.array([2], dtype=np.float32))
x3 = Variable(np.array([3], dtype=np.float32))

"""
x1 = Variable(np.array([1]).astype(np.float32))
x2 = Variable(np.array([2]).astype(np.float32))
x3 = Variable(np.array([3]).astype(np.float32))
"""

z = (x1-2*x2-1)**2+(x2*x3-1)**2+1
z.data


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2)
        )

    def __call__(self, x, y):
        fv = self.fwd(x, y)
        loss = F.mean_squared_error(fv, y)
        return loss

    def fwd(self, x, y):
        return F.sigmoid(self.l1(x))




model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

model.zerograds()
loss = model(x, y)
loss.backward()
optimizer.update()
