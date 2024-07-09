import numpy as np

import torch
import torch.nn as nn



# word embedding layer
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id):
        super(TokenEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output



# positional encoding (embedding) layer
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, pos_encoding, device):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.pos_encoding = pos_encoding
        self.device = device

        self.pos = torch.arange(0, self.max_len)
        if self.pos_encoding:
            self.pe = torch.zeros(self.max_len, self.hidden_dim)
            for i in range(0, self.hidden_dim, 2):
                self.pe[:, i] = np.sin(self.pos/(10000**(i/self.hidden_dim)))
                self.pe[:, i+1] = np.cos(self.pos/(10000**(i/self.hidden_dim)))         
            self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)
        else:
            self.emb_layer = nn.Embedding(self.max_len, self.hidden_dim)


    def forward(self, x):
        if self.pos_encoding:
            return self.pe[:, :x.size(1)]
        return self.emb_layer(self.pos.unsqueeze(0).to(self.device))[:, :x.size(1)]