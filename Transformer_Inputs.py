import torch.nn as nn
import torch
import math
from torch.autograd import Variable

#word Embedding layer
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        # d_model —— the demension of the embedding
        # vocab —— the scale of the dictionary
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model

    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)
'''
embedding=nn.Embedding(10,3)
input1=torch.LongTensor([[1,2,4,5],[9,2,5,6]])
print(embedding(input1))

embedding=nn.Embedding(10,3,padding_idx=0)
input1=torch.LongTensor([[1,2,4,5],[9,2,5,6]])
print(embedding(input1))'''

#position encoder
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)

        # position encode
        pe=torch.zeros(max_len,d_model)

        # absolute position
        position=torch.arange(0,max_len).unsqueeze(1)

        # we want to transfer the position to pe, need a 1*d_model matrix
        div_terms=torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position * div_terms)
        pe[:,1::2]=torch.cos(position * div_terms)
        pe=pe.unsqueeze(0)

        # unable to change the pe while optimization,due to the pe is only an initialized matrix
        # able to reload it when you save and reload the module
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)




"""d_model=512
vocab=1000
dropout=0.1
max_len=60

x=Variable(torch.LongTensor([[1,2,3,4],[29,192,21,98]]))

emb=Embeddings(d_model,vocab)
embr=emb(x)
print(embr)
print(embr.shape)

pe=PositionalEncoding(d_model,dropout,max_len)
pe_result=pe(embr)
print('pe_result:',pe_result)"""