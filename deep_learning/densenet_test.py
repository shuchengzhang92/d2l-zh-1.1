import d2lzh as d2l
from mxnet import gluon,init,nd
from mxnet.gluon import nn


def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=3,padding=1))
    return blk

# 稠密块
class DenseBlock(nn.Block):
    def __init__(self,num_convs,num_channels,**kwargs):
        super(DenseBlock,self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = nd.concat(x,y,dim=1)
        return x


# 过渡层
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=1),
            nn.AvgPool2D(pool_size=2,strides=2))
    return blk



net = nn.Sequential()
net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
        nn.BatchNorm(),nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1))

num_channels,growth_rate = 64,32
num_convs_in_dense_blocks = [4,4,4,4]
for i,num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs,growth_rate))
    num_channels +=num_convs*growth_rate
    if i !=len(num_convs_in_dense_blocks)-1:
        num_channels //=2
        net.add(transition_block(num_channels))


net.add(nn.BatchNorm(),nn.Activation('relu'),nn.GlobalAvgPool2D(),
        nn.Dense(10))


lr,num_epochs,batch_size,ctx = 0.1,5,128,d2l.try_gpu()
net.initialize(ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)
d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
