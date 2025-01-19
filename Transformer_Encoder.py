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
        scores=scores.maskedfill(mask==0, -1e9)

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

        self.d_k=embedding_dim % head
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
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head,self.d_k)

        return self.linears[-1](x)


head=8
embedding_dim=512

#
d_model=512
vocab=1000
dropout=0.1
max_len=60

x=Variable(torch.LongTensor([[1,2,3,4],[29,192,21,98]]))
emb=Transformer_Inputs.Embeddings(d_model,vocab)
embr=emb(x)
pe=Transformer_Inputs.PositionalEncoding(d_model,dropout,max_len)
pe_result=pe(embr)
#

query=key=value=pe_result
mask=Variable(torch.zeros(8,4,4))

mha=MultiHeadAttention(head,embedding_dim,dropout)
mha_result=mha(query,key,value,mask)
print(mha)
print(mha_result.shape)