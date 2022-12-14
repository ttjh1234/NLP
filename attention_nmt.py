from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import re
import random

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20 

class Lang:
    def __init__(self):
        self.word2index={}
        self.word2count={}
        self.index2word={0:"SOS",1:"EOS"}
        self.n_words= 2 
        
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalizeString(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace('[^0-9A-Za-z가-힣\s:]+', ' ')

    return sentence

def read_sentence(df, lang1 ,lang2):
    sentence1=normalizeString(df,lang1)
    sentence2=normalizeString(df,lang2)
    return sentence1, sentence2

def read_file(loc,lang1,lang2):
    df = pd.read_csv(loc,delimiter='\t',header=None,names=[lang1,lang2,'attribute'])
    return df

def process_data(lang1, lang2):
    df = read_file(r'C:\Users\UOS\Desktop\data\kor-eng\kor.txt','eng','kor')
    sentence1, sentence2 =read_sentence(df,lang1,lang2)
    
    sentence1=sentence1.map(lambda x : x.strip())
    sentence2=sentence2.map(lambda x : x.strip())
    
    input_lang = Lang()
    output_lang = Lang()
    pairs = []
    for i in range(len(df)):
        if (len(sentence1[i].split(' ')) < MAX_LENGTH) & (len(sentence2[i].split(' ')) < MAX_LENGTH):
            full=[sentence1[i],sentence2[i]]
            input_lang.addSentence(sentence1[i])
            output_lang.addSentence(sentence2[i])
            pairs.append(full)
            
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

def tensorsFromPair(input_lang,output_lang,pair):
    input_tensor=tensorFromSentence(input_lang,pair[0])
    output_tensor=tensorFromSentence(output_lang,pair[1])
    return(input_tensor,output_tensor)



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim,num_layers,device):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(self.input_dim,self.embed_dim).to(device)
        self.gru=nn.GRU(self.embed_dim,self.hidden_dim,num_layers=self.num_layers).to(device)

    def forward(self,src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        
        # outputs : [sentence_length, batch, hidden_dim]
        # hidden : [n_layer, batch, hidden_dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim,hidden_dim,embed_dim,num_layers,device):
        super().__init__()
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        
        self.embedding=nn.Embedding(self.output_dim, self.embed_dim).to(device)
        self.gru= nn.GRU(self.embed_dim,self.hidden_dim,self.num_layers).to(device)
        self.out = nn.Linear(self.hidden_dim, output_dim).to(device)
        self.softmax = nn.LogSoftmax(dim=1).to(device)

    def forward(self, input, hidden):
        input = input.view(1,-1)
        embed = self.embedding(input)
        output, hidden = self.gru(embed,hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device,MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.device=device
        
    def forward(self,input_lang,output_lang,teacher_forcing_ratio=0.5):
        
        batch_size=output_lang.shape[1]
        target_length= output_lang.shape[0]
        vocab_size=self.decoder.output_dim
        outputs= torch.zeros(target_length,batch_size,vocab_size).to(self.device)
        

        _ , encoder_hidden =encoder(input_lang)
        decoder_hidden=encoder_hidden.to(self.device)
        decoder_input=torch.tensor([SOS_token],device=self.device)
        
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden)
            outputs[t]=decoder_output
            teacher_force=random.random() < teacher_forcing_ratio
            _ , topi = decoder_output.topk(1)
            decoder_input = (output_lang[t] if teacher_force else topi)
            if (teacher_force == False) & (decoder_input.item() == EOS_token):
                break
        return outputs


def Model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    
    loss=0
    epoch_loss=0
    output = model(input_tensor,target_tensor)
    num_iter=output.size(0)
    
    for ot in range(num_iter):
        loss +=criterion(output[ot],target_tensor[ot])
    loss.backward()
    model_optimizer.step()
    epoch_loss=loss.item()/num_iter
    return epoch_loss


def trainModel(model, input_lang,output_lang,pairs,num_iteration=20000):
    model.train()
    optimizer= optim.Adam(model.parameters())
    criterion=nn.NLLLoss()
    total_loss_iterations=0
    
    training_pairs=[tensorsFromPair(input_lang,output_lang,random.choice(pairs)) for _ in range(num_iteration)]


    for iter in range(1, num_iteration+1):
        training_pair=training_pairs[iter-1]
        input_tensor=training_pair[0]
        target_tensor=training_pair[1]
        loss=Model(model,input_tensor,target_tensor,optimizer,criterion)
        total_loss_iterations+=loss
        
        if iter % 100 == 0 :
            average_loss = total_loss_iterations / 100
            total_loss_iterations=0
            print('%d %.4f' % (iter,average_loss))
    
    torch.save(model.state_dict(),r'C:\Users\UOS\Desktop\Sungsu\github\NLP\model\seq2seq.pt')

    return model

def evaluate(model, input_lang, output_lang, sentences):
    #model.eval()
    with torch.no_grad():
        input_tensor=tensorFromSentence(input_lang,sentences[0])
        output_tensor=tensorFromSentence(output_lang,sentences[1])
        decoded_words=[]
                
        output=model(input_tensor,output_tensor)
        
        #print(output)
        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            
            if topi[0].item()==EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])

    return decoded_words

def evaluate_randomly(model,input_lang,output_lang,pairs,n=10):
    for i in range(n):
        pair=random.choice(pairs)
        print('input {}'.format(pair[0]))
        print('output {}'.format(pair[1]))
        output_words=evaluate(model,input_lang,output_lang,pair)
        output_sentence=' '.join(output_words)
        print('predicted {}'.format(output_sentence))


def train_and_evaluate():
    lang1= 'eng'
    lang2= 'kor'
    teacher_forcing_ratio = 0.5


    input_lang, output_lang, pairs = process_data(lang1,lang2)

    randomize = random.choice(pairs)
    print('random sentence : {}'.format(randomize))

    input_size = input_lang.n_words
    output_size = output_lang.n_words
    print('Input : {} Output : {}'.format(input_size,output_size))

    embed_size=256
    hidden_size=512
    num_layers=2
    num_iteration=50000

    encoder=Encoder(input_size, hidden_size , embed_size, num_layers,device=device)
    decoder=Decoder(output_size, hidden_size , embed_size, num_layers,device=device)


    model = Seq2Seq(encoder,decoder,device,teacher_forcing_ratio)

    model = trainModel(model,input_lang,output_lang,pairs,num_iteration)

    evaluate_randomly(model,input_lang,output_lang,pairs,n=20)


class Attention(nn.Module):
    def __init__(self, hidden_size,device):
        super().__init__()
        self.hidden_size=hidden_size
        self.device= device
        
        self.attn=nn.Linear(self.hidden_size *2,self.hidden_size).to(self.device)
        self.v = nn.Linear(self.hidden_size,1,bias=False).to(self.device)
        
        
    def forward(self,hidden, encoder_outputs):

        src_len = encoder_outputs.shape[0]
        hidden=hidden.repeat(1,src_len,1)
        
        encoder_outputs= encoder_outputs.permute(1,0,2)
        energy=torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        attention=self.v(energy).squeeze(2)
        
        attention_score = F.softmax(attention,dim=1).unsqueeze(1)

        return attention_score


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, device, attention, max_length=MAX_LENGTH):
        super().__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.max_length=max_length
        self.device= device
        self.attention = attention
        
        self.embedding=nn.Embedding(self.output_size,self.hidden_size).to(self.device)       
        self.gru= nn.GRU(self.hidden_size*2,self.hidden_size).to(self.device)
        self.out= nn.Linear(self.hidden_size,self.output_size).to(self.device)
        self.softmax = nn.LogSoftmax(dim=1).to(self.device)
        
    def forward(self,input, hidden, encoder_outputs):
        embed = self.embedding(input).view(1,1,-1)
        
        attention_score = self.attention(hidden, encoder_outputs)
        encoder_outputs=encoder_outputs.permute(1,0,2)
        
        weighted=torch.bmm(attention_score,encoder_outputs)

        # 1, 1, 1024
        gru_input = torch.cat((embed,weighted),dim=2)
        
        output, hidden = self.gru(gru_input,hidden)
        prediction = self.softmax(self.out(output[0]))
        
        return prediction, hidden
        
class AttSeq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device):
        super().__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.device=device
        
    def forward(self,input_lang,output_lang,teacher_forcing_ratio=0.5):
        
        batch_size=output_lang.shape[1]
        target_length= output_lang.shape[0]
        vocab_size=self.decoder.output_size
        outputs= torch.zeros(target_length,batch_size,vocab_size).to(self.device)
        

        encoder_outputs , encoder_hidden = self.encoder(input_lang)
        
        decoder_hidden=encoder_hidden.to(self.device)
        decoder_input=torch.tensor([SOS_token],device=self.device)
        
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            outputs[t]=decoder_output
            teacher_force=random.random() < teacher_forcing_ratio
            _ , topi = decoder_output.topk(1)
            decoder_input = (output_lang[t] if teacher_force else topi)
            if (teacher_force == False) & (decoder_input.item() == EOS_token):
                break
        return outputs


lang1= 'eng'
lang2= 'kor'
teacher_forcing_ratio = 0.5


input_lang, output_lang, pairs = process_data(lang1,lang2)

randomize = random.choice(pairs)
print('random sentence : {}'.format(randomize))

input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size,output_size))

embed_size=256
hidden_size=512
num_layers=1
num_iteration=50000

encoder=Encoder(input_size, hidden_size , embed_size, num_layers,device=device)
attention=Attention(hidden_size,device)
decoder=AttentionDecoder(hidden_size, output_size, device, attention)
model = AttSeq2Seq(encoder,decoder,device)

model = trainModel(model,input_lang,output_lang,pairs,num_iteration)

evaluate_randomly(model,input_lang,output_lang,pairs,n=20)


encoder=Encoder











