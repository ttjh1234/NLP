import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import BertModel


model=BertModel.from_pretrained("bert-base-uncased")


model.num_parameters()
print(model)


class BertV2(nn.module):
    def __init__(self,):
        super().__init__()
        
    def forward(self,input):
        out=None
        return out


class embedding_v2(nn.module):
    def __init__(self,):
        super().__init__()
        
    def forward(self,input):
        out=None
        return out


class Attention_bert(nn.module):
    def __init__(self,):
        super().__init__()
        
    def forward(self,input):
        out=None
        return out




