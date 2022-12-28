# Use : IMDB Datasets
# For : Solve Sentence Classification Problem
# But, Implement Encoder - Decoder Structure. (Later other project using it.)

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import IMDB

import numpy as np
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 500):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim,dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        if src.dim==2:
            batch_size = src.shape[0]
            src_len = src.shape[1]
        
            #pos = [batch size, src len]
            pos = torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(device)
        
            # src = [batch size, src len, hid dim]
            # src: input embedding에 scale을 곱해주고 positional encoding 더함.
            src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        else:
            batch_size = src.shape[0]
            src_len = src.shape[1]
        
            #pos = [batch size, src len]
            pos = torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(device)
        
            # src = [batch size, src len, hid dim]
            # src: input embedding에 scale을 곱해주고 positional encoding 더함.
            src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        
        for layer in self.layers:
          src = layer(src, src_mask)
          
        #src = [batch size, src len, hid dim]
        return src


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        Q = self.fc_q(query) # src
        K = self.fc_k(key) # src
        V = self.fc_v(value) # src
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy,dim=-1)
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]
        
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]
        
        return x


class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).to(device)
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        #output = [batch size, trg len, output dim]
            
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #decoder-encoder cross attention
        _trg, attention = self.encoder_attention(trg,enc_src,enc_src,src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
    
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):        
        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]

        # torch.tril: The lower triangular part of the matrix is defined as the elements on and below the diagonal.
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    

class sentiment_classification(nn.Module):
    def __init__(self,encoder,hidden_dim,src_pad_idx,device,output_dim=1):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.src_pad_idx = src_pad_idx
        self.device=device
        self.linear1 = nn.Linear(self.hidden_dim,self.hidden_dim//2)
        self.linear2 = nn.Linear(self.hidden_dim//2,self.output_dim)
        self.sigmoid= nn.Sigmoid()
    
    def make_src_mask(self, src):
        #src = [batch size, src len]
        if src.dim==2:        
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
            #src_mask = [batch size, 1, 1, src len]
        else:
            n_batch=src.shape[0]
            n_length=src.shape[1]
            n_emb_length=src.shape[2]
            src_mask = (torch.sum(torch.eq(torch.flatten(src,0,1),src[0][0]),dim=1)==n_emb_length).view(n_batch,n_length).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def forward(self,input):
        
        src_mask = self.make_src_mask(input)
        enc_src = self.encoder(input, src_mask)
        # [Batch size, input len, hidden dim]
        out1=self.linear1(enc_src[:,0,:])
        out=self.linear2(out1)
        out=self.sigmoid(out)
        return out
        
        

def imdb_pytorch_load(vocab_size=10000,max_len=500,batch_size=64):
    TEXT = torchtext.data.Field(sequential=True, lower=True, batch_first=True,fix_length=max_len)
    LABEL = torchtext.data.Field(sequential=False,batch_first= True)
    trainset, testset = torchtext.datasets.IMDB.splits(TEXT, LABEL)
    
    TEXT.build_vocab(trainset, min_freq=5,max_size=vocab_size) # 단어 집합 생성
    LABEL.build_vocab(trainset)
    
    vocab_size = len(TEXT.vocab)
    n_classes = 2
    print('단어 집합의 크기 : {}'.format(vocab_size))
    print('클래스의 개수 : {}'.format(n_classes))

    trainset, valset = trainset.split(split_ratio=0.8)
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=batch_size, shuffle=True, repeat=False)
    
    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
    print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))
    
    input_dim=len(TEXT.vocab)
    TEXT_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    return train_iter, val_iter, test_iter, input_dim, TEXT_PAD_IDX, TEXT


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for _, batch in enumerate(iterator):
        
        optimizer.zero_grad()
        
        src = batch.text.to(device)
        target = batch.label.to(device).to(torch.float32) - 1
        
        output = model(src)
            
        output=output.view(-1)
        output = output.contiguous()
            
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for _, batch in enumerate(iterator):

            src = batch.text.to(device)
            trg = batch.label.to(device).to(torch.float32) - 1
            output = model(src)
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output=output.view(-1)
            output = output.contiguous()
        
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def accuracy_score(model,train,valid,test):
    model.eval()
    
    total_train=0
    train_acc=0
    total_valid=0
    valid_acc=0
    total_test=0
    test_acc=0
        
    with torch.no_grad():
        for _, batch in enumerate(train):
            total_train+=batch.text.shape[0]
            src = batch.text.to(device)
            trg = batch.label.to(device).to(torch.float32) - 1
            output = model(src)
            output=output.view(-1)
            output = torch.where(output>=0.5,1,0)
            train_acc+=torch.sum(torch.where(output==trg,1,0))
    
    train_accuracy=train_acc/total_train

    with torch.no_grad():
        for _, batch in enumerate(valid):
            total_valid+=batch.text.shape[0]
            src = batch.text.to(device)
            trg = batch.label.to(device).to(torch.float32) - 1
            output = model(src)
            output=output.view(-1)
            output = torch.where(output>=0.5,1,0)
            valid_acc+=torch.sum(torch.where(output==trg,1,0))
    
    valid_accuracy=valid_acc/total_valid
    
    with torch.no_grad():
        for _, batch in enumerate(test):
            total_test+=batch.text.shape[0]
            src = batch.text.to(device)
            trg = batch.label.to(device).to(torch.float32) - 1
            output = model(src)
            output=output.view(-1)
            output = torch.where(output>=0.5,1,0)
            test_acc+=torch.sum(torch.where(output==trg,1,0))
    
    test_accuracy=test_acc/total_test

    print("--------------------------------------------")
    print("Train Accuracy : ",train_accuracy)
    print("Valid Accuracy : ",valid_accuracy)
    print("Test Accuracy : ",test_accuracy)
    print("--------------------------------------------")


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

