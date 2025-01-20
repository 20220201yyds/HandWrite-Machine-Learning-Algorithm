import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy
import Transformer_Inputs

# generate mask tensor: for the reason that the input should not include the information from the future
def subsequent_mask(size):
    attn_shape=(1,size,size)

    # Upper triangular matrix
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')

    return torch.from_numpy(1-subsequent_mask)


def attention(query,key,value,mask=None,dropout=None):
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        print(scores.shape,mask.shape)
        scores=scores.masked_fill(mask==0, -1e9)
    p_attn=F.softmax(scores,dim=-1)

    if dropout is not None:
        p_attn=dropout(p_attn)

    return torch.matmul(p_attn,value),p_attn


# for multi_head_attention, we have to copy the same structure head
# also the copies need be stored at different memory space
def clones(module,N):
    # N:duplicate time
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self,head,embedding_dim,dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % head == 0

        self.d_k=embedding_dim // head
        self.head=head
        # 4 for q,k,v,result
        self.linears=clones(nn.Linear(embedding_dim,embedding_dim),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)

        batch_size=query.size(0)

        query,key,value=\
            [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
             for model,x in zip(self.linears,(query,key,value))]

        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)

        # contiguous() is necessary for view in torch
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head * self.d_k)

        return self.linears[-1](x)

# the experiment shows that add 2 FeedForward layer can improve the model, the dimension wont change
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a2=nn.Parameter(torch.ones(features))
        self.b2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a2*(x-mean)/(std+self.eps)+self.b2

# Residual Connection
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
#
d_model=512
vocab=1000
dropout=0.1
max_len=60

x=Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
emb=Transformer_Inputs.Embeddings(d_model,vocab)
embr=emb(x)
# PositionalEncoding result
pe=Transformer_Inputs.PositionalEncoding(d_model,dropout,max_len)
pe_result=pe(embr)


# MultiHeadAttention result
head=8
embedding_dim=512
query=key=value=pe_result
mask=Variable(torch.zeros(2,4,4))
#attn,pttn=attention(query,key,value,mask)
#print(attn,pttn)
mha=MultiHeadAttention(head,embedding_dim,dropout)
mha_result=mha(query,key,value,mask)
print('mha',mha_result)
print(mha_result.shape)

# PositionwiseFeedForward result
d_ff=64
ff=PositionwiseFeedForward(d_model,d_ff,dropout)
ff_result=ff(mha_result)

# Norm result
feature=d_model
ln=LayerNorm(feature)
ln_result=ln(ff_result)
print('ln_result',ln_result)

x=pe_result
self_attn=MultiHeadAttention(head,d_model)
sublayer=lambda x:self_attn(x,x,x,mask)
sc=SublayerConnection(512,dropout)
sc_result=sc(pe_result,sublayer)
print('sc_result',sc_result)