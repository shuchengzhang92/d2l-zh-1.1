from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Dense(256,activation='relu') #隐藏层
        self.output = nn.Dense(10) #输出层

    def forward(self, x):
        return self.output(self.hidden(x))


class MySequential(nn.Block):
    def __init__(self,**kwargs):
        super(MySequential,self).__init__(**kwargs)


    def add(self,block):
        print(block.name)
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x



class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant('rand_weight',nd.random.uniform(shape=(20,20)))
        self.dense = nn.Dense(20,activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x,self.rand_weight.data()) +1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /=2
        if x.norm().asscalar() <0.8:
            x *=10
        return x.sum()



x = nd.random.uniform(shape=(2,20))
net = MySequential()
net.add(nn.Dense(256,activation='relu'))
net.add(nn.Dense(10))
net.initialize()
y = net(x)










#
# X = nd.random.uniform(shape=(2,20))
# net = MLP()
# net.initialize()
# y = net(X)
# print(y)