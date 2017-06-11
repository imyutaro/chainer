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
    def __call__(self, x):
        h = self.l2(F.sigmoid(self.l1(x)))
        return h

# Sub routine

## Utility: Summarize current results
def summarize(model, optimizer, inputs, outputs):
    print('model says:')
    for i in range(len(inputs)):
        x  = Variable(inputs[i].reshape(1,2).astype(np.float32))
        t  = Variable(outputs[i].astype(np.int32))
        y = model.predictor(x)
        print('  %d & %d = %d (zero:%f one:%f)' % (x.data[0,0], x.data[0,1], np.argmax(y.data), y.data[0,0], y.data[0,1]))
    #mean_loss = sum_loss / len(inputs)
    #mean_accuracy = sum_accuracy / len(inputs)
    #print sum_loss, sum_accuracy, mean_loss, mean_accuracy

## Runs learning loop
def learning_looper(model, optimizer, inputs, outputs, epoch_size):
    augment_size = 100
    for epoch in range(epoch_size):
        print('epoch %d' % epoch)
        for a in range(augment_size):
            for i in range(len(inputs)):
                x = Variable(inputs[i].reshape(1,2).astype(np.float32))
                t = Variable(outputs[i].astype(np.int32))
                model.zerograds()
                loss = model(x, t)
                loss.backward()
                optimizer.update()
                #optimizer.update(model, x, t)
        summarize(model, optimizer, inputs, outputs)

# Main

## Test data
inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
and_outputs = np.array([[0], [0], [0], [1]], dtype=np.int32)
or_outputs = np.array([[0], [1], [1], [1]], dtype=np.int32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.int32)

## AND Test --> will learn successfully
and_model = L.Classifier(NN2x2_2links())
optimizer = optimizers.SGD()
# do it quicker) optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(and_model)
print('<<AND: Before learning>>')
summarize(and_model, optimizer, inputs, and_outputs)
print('\n<<AND: After Learning>>')
learning_looper(and_model, optimizer, inputs, and_outputs, epoch_size = 20)

## OR Test --> will learn successfully
or_model = L.Classifier(NN2x2_2links())
optimizer = optimizers.SGD()
optimizer.setup(or_model)
print('\n---------\n\n<<OR: Before learning>>')
summarize(or_model, optimizer, inputs, or_outputs)
print('\n<<OR: After Learning>>')
learning_looper(or_model, optimizer, inputs, or_outputs, epoch_size = 20)

## XOR Test --> will learn successfully
xor_model = L.Classifier(NN2x2_2links())
optimizer = optimizers.SGD()
optimizer.setup(xor_model)
print('\n---------\n\n<<XOR: Before learning>>')
summarize(xor_model, optimizer, inputs, xor_outputs)
print('\n<<XOR: After Learning>>')
learning_looper(xor_model, optimizer, inputs, xor_outputs, epoch_size = 200)
