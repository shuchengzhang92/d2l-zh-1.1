import d2lzh as d2l
from mxnet import gluon,init,nd
from mxnet.gluon import nn

class Residual(nn.Block):
    def __init__(self,num_channels,use_1x1conv=False,strides =1,**kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels,kernel_size=3,padding=1,strides=strides)
        self.conv2 = nn.Conv2D(num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        y = nd.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return nd.relu(y+x)

net = nn.Sequential()
net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
        nn.BatchNorm(),nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1))
def resnet_block(num_channels,num_residuals,first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels,use_1x1conv=True,strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


net.add(resnet_block(64,2,first_block=True),resnet_block(128,2),resnet_block(256,2),resnet_block(512,2))
net.add(nn.GlobalAvgPool2D(),nn.Dense(10))
lr,num_epochs,batch_size,ctx = 0.05,5,256,d2l.try_gpu()
net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)
d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
