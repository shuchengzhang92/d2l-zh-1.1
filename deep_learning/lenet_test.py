import d2lzh as d2l
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10))


# x = nd.random.uniform(shape=(1,1,28,28))
# net.initialize()
# for layer in net:
#     x= layer(x)
#     print(layer.name,x.shape)
#

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu()
print(ctx)

def evaluate_accuracy(data_iter,net,ctx):
    acc_sum,n = nd.array([0],ctx=ctx),0
    for x,y in data_iter:
        x,y = x.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
        acc_sum += (net(x).argmax(axis=1)==y).sum()
        n +=y.size
    return acc_sum.asscalar()/n




def train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs):
    print(ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()
        for x,y in train_iter:
            x,y = x.as_in_context(ctx),y.as_in_context(ctx)
            # print(x,y)
            with autograd.record():
                y_hat = net(x)
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            # print((y_hat.argmax(axis=1)==y).sum())
            train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
            n +=y.size
        test_acc = evaluate_accuracy(test_iter,net,ctx)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f,'
              'time %.1f sec'
              %(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc,time.time()-start))



lr,num_epochs =0.9,5
default_init=mx.initializer.Uniform()
net.initialize(ctx=ctx,init=default_init)
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)


