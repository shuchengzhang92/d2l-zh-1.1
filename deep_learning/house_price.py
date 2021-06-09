import d2lzh as d2l

from mxnet import autograd,gluon,init,nd
from mxnet.gluon import data as gdata,loss as gloss,nn
import numpy as np
import pandas as pd
# 读取数据集
train_data = pd.read_csv('../data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('../data/kaggle_house_pred_test.csv')
print(train_data.shape)
print(test_data.shape)
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
print(all_features.shape)
# 预处理数据集
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
print(numeric_features)
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape)
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
print(len(train_features))
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1,1))
print(train_labels)
# 训练模型
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net


def log_rmse(net,features,labels):
    clipped_preds = nd.clip(net(features),1,float('inf'))
    rmse = nd.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
    return rmse.asscalar()


def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':learning_rate,'wd':weight_decay})
    for epoch in range(num_epochs):
        for X ,y in train_iter:
            with autograd.record():
                l = loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls


def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0] //k
    x_train ,y_train = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part = X[idx,:],y[idx]
        if j ==i:
            x_valid ,y_valid = x_part,y_part
        elif x_train is None:
            x_train ,y_train = x_part,y_part
        else:
            x_train = nd.concat(x_train,x_part,dim=0)
            y_train = nd.concat(y_train,y_part,dim=0)
    return x_train,y_train,x_valid,y_valid


def k_fold(k,x_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum = 0,0
    for i in range(k):
        data = get_k_fold_data(k,i,x_train,y_train)
        print(data)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum +=train_ls[-1]
        valid_l_sum +=valid_ls[-1]
        if i ==0:
            d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse',range(1,num_epochs+1),valid_ls,['train','valid'])
        print('fold %d,train rmse %f,valid rmse %f' %(i,train_ls[-1],valid_ls[-1]))
    return train_l_sum/k,valid_l_sum/k



k,num_epochs,learning_rate,weight_decay,batch_size = 5,100,5,0,64
train_l ,valid_l = k_fold(k,train_features,train_labels,num_epochs,learning_rate,weight_decay,batch_size)
print(k,train_l,valid_l)