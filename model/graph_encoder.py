from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn
import numpy as np

def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t



class multi_head_attention(torch.nn.Module):
    def __init__(self,d_key,d_value,d_model,n_head,attention_dropout):
        super(multi_head_attention,self).__init__()
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.n_head=n_head
        self.attention_dropout=attention_dropout

        self.layer_q=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
        self.layer_q.weight.data=truncated_normal(self.layer_q.weight.data,std=0.02)
        #torch.nn.init.xavier_uniform_(self.layer_q.weight)
        torch.nn.init.constant_(self.layer_q.bias, 0.0)
        self.layer_k=torch.nn.Linear(self.d_model,self.d_key * self.n_head)
        self.layer_k.weight.data=truncated_normal(self.layer_k.weight.data,std=0.02)
        #torch.nn.init.xavier_uniform_(self.layer_k.weight)
        torch.nn.init.constant_(self.layer_k.bias, 0.0)
        self.layer_v=torch.nn.Linear(self.d_model,self.d_value * self.n_head)
        self.layer_v.weight.data=truncated_normal(self.layer_v.weight.data,std=0.02)
        #torch.nn.init.xavier_uniform_(self.layer_v.weight)
        torch.nn.init.constant_(self.layer_v.bias, 0.0)
        self.project_layer=torch.nn.Linear(d_value * n_head,self.d_model)
        self.project_layer.weight.data=truncated_normal(self.project_layer.weight.data,std=0.02)
        #torch.nn.init.xavier_uniform_(self.project_layer.weight)
        torch.nn.init.constant_(self.project_layer.bias, 0.0)

    def forward(self,
                queries,
                edges_key,
                edges_value,
                attn_bias):
        #B is batch_size, M is max_seq_len, N is n_head, H is d_key
        batch_size=queries.size(0)
        max_seq_len=queries.size(1)
        #query,key,value is [B,M,N*H], edges_key,edges_value is [M,M,H], attn_bias is [B,N,M,M]
        keys = queries 
        values = keys 
        #q,k,v is [B,N,M,H]
        q=self.layer_q(queries).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2)         
        k=self.layer_k(keys).view(batch_size,-1,self.n_head,self.d_key).transpose(1,2)            
        v=self.layer_v(values).view(batch_size,-1,self.n_head,self.d_value).transpose(1,2) 
        #scores1,scores2,scores is [B,N,M,M]
        scores1 = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_key)
        scores2 = torch.matmul(q.permute(2,0,1,3).contiguous().view(max_seq_len,-1,self.d_key),edges_key.transpose(-1,-2)).view(max_seq_len,-1,self.n_head,max_seq_len).permute(1,2,0,3)/ np.sqrt(self.d_key)
        scores=torch.add(scores1,scores2)
        scores=torch.add(scores,attn_bias)
        #weights is [B,N,M,M]
        weights=torch.nn.Dropout(self.attention_dropout)(torch.nn.Softmax(dim=-1)(scores))
        #context1,context2,context is [B,N,M,H]
        context1= torch.matmul(weights,v)
        context2= torch.matmul(weights.permute(2,0,1,3).contiguous().view(max_seq_len,-1,max_seq_len),edges_value).view(max_seq_len,-1,self.n_head,self.d_value).permute(1,2,0,3)
        context=torch.add(context1,context2)
        #output is [B,M,N*H]
        output=context.transpose(1,2).contiguous().view(batch_size,-1,self.n_head*self.d_value)
        output=self.project_layer(output)
        return output


class positionwise_feed_forward(torch.nn.Module):
    def __init__(self,d_inner_hid,d_model):
        super(positionwise_feed_forward,self).__init__()
        self.d_inner_hid=d_inner_hid
        self.d_hid=d_model

        self.fc1=torch.nn.Linear(self.d_hid,self.d_inner_hid)
        self.fc1.weight.data=truncated_normal(self.fc1.weight.data,std=0.02)
        #torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2=torch.nn.Linear(self.d_inner_hid,self.d_hid)
        self.fc2.weight.data=truncated_normal(self.fc2.weight.data,std=0.02)
        #torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self,x):       
        return self.fc2(torch.nn.GELU()(self.fc1(x)))

class encoder_layer(torch.nn.Module):
    def __init__(self,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout):
        super(encoder_layer,self).__init__()
        self.n_head=n_head
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.d_inner_hid=d_inner_hid
        self.prepostprocess_dropout=prepostprocess_dropout
        self.attention_dropout=attention_dropout

        self.multi_head_attention=multi_head_attention(
            self.d_key,
            self.d_value,
            self.d_model,
            self.n_head,
            self.attention_dropout)
        self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self.d_model,eps=1e-7,elementwise_affine=True)

        self.positionwise_feed_forward=positionwise_feed_forward(
            self.d_inner_hid,
            self.d_model)
        self.layer_norm2=torch.nn.LayerNorm(normalized_shape=self.d_model,eps=1e-7,elementwise_affine=True)

    def forward(self,enc_input,
                    edges_key,
                    edges_value,
                    attn_bias):
        attn_output = self.multi_head_attention(
            enc_input,
            edges_key,
            edges_value,
            attn_bias)
        attn_output=self.layer_norm1(torch.add(enc_input,torch.nn.Dropout(self.prepostprocess_dropout)(attn_output)))
        
        ffd_output = self.positionwise_feed_forward(attn_output)
        ffd_output=self.layer_norm2(torch.add(attn_output,torch.nn.Dropout(self.prepostprocess_dropout)(ffd_output)))
        return ffd_output


class encoder(torch.nn.Module):
    def __init__(self,n_layer,n_head,d_key,d_value,d_model,
                d_inner_hid,prepostprocess_dropout,attention_dropout):
        super(encoder,self).__init__()
        self.n_layer=n_layer
        self.n_head=n_head
        self.d_key=d_key
        self.d_value=d_value
        self.d_model=d_model
        self.d_inner_hid=d_inner_hid
        self.prepostprocess_dropout=prepostprocess_dropout
        self.attention_dropout=attention_dropout

        for nl in range(self.n_layer):        
            setattr(self,"encoder_layer{}".format(nl),encoder_layer(
                self.n_head,
                self.d_key,
                self.d_value,
                self.d_model,
                self.d_inner_hid,
                self.prepostprocess_dropout,
                self.attention_dropout))

    def forward(self,enc_input,edges_key,edges_value,attn_bias):
        for nl in range(self.n_layer):
            enc_output = getattr(self,"encoder_layer{}".format(nl))(
                enc_input,
                edges_key,
                edges_value,
                attn_bias)
            enc_input = enc_output
        return enc_output
