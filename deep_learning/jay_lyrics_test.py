import d2lzh as d2l
import math
from mxnet import autograd,nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices,char_to_idx,idx_to_char,vocab_size) = d2l.load_data_jay_lyrics()

# nd.one_hot(nd.array([0,2]),vocab_size)
# 初始化模型参数
num_inputs,num_hiddens,num_outputs = vocab_size,256,vocab_size
ctx = d2l.try_gpu()


def to_onehot(x,size):
    return [nd.one_hot(x,size) for x in x.T]

x = nd.arange(10).reshape((2,5))
inputs = to_onehot(x,vocab_size)
def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01,shape=shape,ctx=ctx)
    w_xh = _one((num_inputs,num_hiddens))
    w_hh = _one((num_hiddens,num_hiddens))
    b_h = nd.zeros(num_hiddens,ctx=ctx)
    w_hq = _one((num_hiddens,num_outputs))
    b_q = nd.zeros(num_outputs,ctx=ctx)
    params = [w_xh,w_hh,b_h,w_hq,b_q]
    for param in params:
        param.attach_grad()
    return params


# 定义模型
def init_rnn_state(batch_size,num_hiddens,ctx):
    return (nd.zeros(shape=(batch_size,num_hiddens),ctx=ctx))


def rnn(inputs,state,params):
    w_xh,w_hh,b_h,w_hq,b_q = params
    h= state
    outputs = []
    for x in inputs:
        h = nd.tanh(nd.dot(x,w_xh)+nd.dot(h,w_hh) + b_h)
        y = nd.dot(h,w_hq)+b_q
        outputs.append(y)
    return outputs,(h,)
state = init_rnn_state(x.shape[0],num_hiddens,ctx)
inputs = to_onehot(x.as_in_context(ctx),vocab_size)
params = get_params()
print(params,'params')
out_puts,state_new = rnn(inputs,state,params)
print(len(out_puts),out_puts[0].shape,state_new[0].shape)


# 定义预测函数
def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,num_hiddens,vocab_size,ctx,idx_to_char,char_to_idx):
    state = init_rnn_state(1,num_hiddens,ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        x = to_onehot(nd.array([output[-1]],ctx=ctx),vocab_size)
        (y,state) = rnn(x,state,params)
        if t < len(prefix) -1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])


result = predict_rnn('分开',10,rnn,params,init_rnn_state,num_hiddens,vocab_size,ctx,idx_to_char,char_to_idx)
print(result)