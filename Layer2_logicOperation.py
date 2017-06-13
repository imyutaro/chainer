# Chainer training: And/Or/Xor classifier network example with 2 links.
#
# This is re-written version of:
#   http://hi-king.hatenablog.com/entry/2015/06/27/194630
# By following chainer introduction:
#   http://docs.chainer.org/en/stable/tutorial/basic.html

## Chainer cliche
import numpy as np
import chainer
from chainer import Function, Variable, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F

import chainer.links as L

# Neural Network

## Network definition
class NN2x2_2links(Chain):
    def __init__(self):
        super(NN2x2_2links, self).__init__(
            l1 = L.Linear(2, 2),
            l2 = L.Linear(2, 2),
        )
    def __call__(self, x, y):
        fv = self.forward(x,y)
        loss = F.mean_squared_error(fv,y)
        return loss

    def forward(self, x):
        return self.l2(F.sigmoid(self.l1(x)))


# Sub routine

## Utility: Summarize current results
"""
def summarize(model, optimizer, inputs, outputs):
    sum_loss, sum_accuracy = 0, 0
    print('model says:')
    for i in range(len(inputs)):
        x  = Variable(inputs[i].reshape(1,2).astype(np.float32))
        t  = Variable(outputs[i].astype(np.int32))
        y = model.predictor(x)
        #loss = model(x, t)
        #sum_loss += loss.data
        #sum_accuracy += model.accuracy.data
        print('  %d & %d = %d (zero:%f one:%f)' % (x.data[0,0], x.data[0,1], np.argmax(y.data), y.data[0,0], y.data[0,1]))
    #mean_loss = sum_loss / len(inputs)
    #mean_accuracy = sum_accuracy / len(inputs)
    #print sum_loss, sum_accuracy, mean_loss, mean_accuracy
"""
## Runs learning loop
def learning_looper(model, optimizer, inputs, outputs, epoch_size):
    augment_size = 100
    for epoch in range(epoch_size):
        print('epoch %d' % epoch)
        for a in range(augment_size):
            for i in range(len(inputs)):
                x = Variable(inputs[i].reshape(1,2).astype(np.float32))
                t = Variable(outputs[i].astype(np.int32))
                #optimizer.update(model, x, t)
                h = model.forward(x)
                model.zerograds()
                error = F.softmax_cross_entropy(h, t)
                accuracy = F.accuracy(h, t)
                error.backward()
                optimizer.update()
        #summarize(model, optimizer, inputs, outputs)
            print('  %d & %d = %d (zero:%f one:%f)' % (x.data[0,0], x.data[0,1], np.argmax(h.data), h.data[0,0], h.data[0,1]))

"""
## Runs XOR_learning loop
def XOR_learning_looper(model, optimizer, inputs, outputs, epoch_size):
    augment_size = 100
    for epoch in range(epoch_size):
        if epoch%10==0:
            print('epoch %d' % epoch)
        for a in range(augment_size):
            for i in range(len(inputs)):
                x = Variable(inputs[i].reshape(1,2).astype(np.float32))
                t = Variable(outputs[i].astype(np.int32))
                optimizer.update(model, x, t)
        if epoch%10==0:
            summarize(model, optimizer, inputs, outputs)
"""
# Main

## Test data
inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
and_outputs = np.array([[0], [0], [0], [1]], dtype=np.int32)
or_outputs = np.array([[0], [1], [1], [1]], dtype=np.int32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.int32)

## AND Test --> will learn successfully
#and_model = L.Classifier(NN2x2_2links())
and_model = NN2x2_2links()
optimizer = optimizers.SGD()
# do it quicker) optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(and_model)
print('<<AND: Before learning>>')
#summarize(and_model, optimizer, inputs, and_outputs)
print('\n<<AND: After Learning>>')
learning_looper(and_model, optimizer, inputs, and_outputs, epoch_size = 21)

## OR Test --> will learn successfully
#or_model = L.Classifier(NN2x2_2links())
or_model = NN2x2_2links()
optimizer = optimizers.SGD()
optimizer.setup(or_model)
print('\n---------\n\n<<OR: Before learning>>')
#summarize(or_model, optimizer, inputs, or_outputs)
print('\n<<OR: After Learning>>')
learning_looper(or_model, optimizer, inputs, or_outputs, epoch_size = 21)

## XOR Test --> will learn successfully
xor_model = L.Classifier(NN2x2_2links())
#optimizer = optimizers.SGD()
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(xor_model)
print('\n---------\n\n<<XOR: Before learning>>')
#summarize(xor_model, optimizer, inputs, xor_outputs)
print('\n<<XOR: After Learning>>')
#XOR_learning_looper(xor_model, optimizer, inputs, xor_outputs, epoch_size = 251)
