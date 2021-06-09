from mxnet import nd
import random
import zipfile


with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])
corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')[:10000]
# 建立字符索引
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char,i) for i,char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size)
corpus_indices = [char_to_idx[char] for char in corpus_chars]
# sample = corpus_chars[:20]


# 随机采样
def data_iter_random(corpus_indices,batch_size,num_steps,ctx=None):
    num_examples = (len(corpus_indices) -1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)


    def _data(pos):
        return corpus_indices[pos:pos+num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i+batch_size]
        x = [_data(j*num_steps) for j in batch_indices]
        y = [_data(j*num_steps)  for j in batch_indices]
        yield  nd.array(x,ctx),nd.array(y,ctx)


# 相邻采样
def data_iter_consecutive(corpus_indices,batch_size,num_steps,ctx=None):
    corpus_indices = nd.array(corpus_indices,ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0:batch_size*batch_len].reshape((batch_size,batch_len))
    epoch_size = (batch_len-1)//num_steps
    for i in range(epoch_size):
        i = i*num_steps
        x = indices[i:i:i+num_steps]
        y = indices[:,i+1:i+num_steps+1]
        yield x,y


