import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


# print(mx.cpu(),mx.gpu())
# print(mx.gpu(1))


a = nd.array([1,2,3],ctx=mx.gpu(0))
print(a)