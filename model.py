import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# word embedding layer
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output



# positional encoding layer
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_dim, pos_encoding, device):
        super(PositionalEncoding, self).__init__()
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
            return self.pe[:, :x.size(1), :]
        return self.emb_layer(self.pos.unsqueeze(0).to(self.device))
        



# mulithead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_head, bias, self_attn, causal):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.bias = bias
        self.self_attn = self_attn
        self.causal = causal
        self.head_dim = self.hidden_dim // self.num_head
        assert self.hidden_dim == self.num_head * self.head_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)


    def head_split(self, x):
        x = x.view(self.batch_size, -1, self.num_head, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x


    def scaled_dot_product(self, q, k, v, mask):
        attn_wts = torch.matmul(q, torch.transpose(k, 2, 3))/math.sqrt(self.hidden_dim)
        if not mask == None:
            attn_wts = attn_wts.masked_fill(mask==0, float('-inf'))
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_out = torch.matmul(attn_wts, v)
        return attn_wts, attn_out


    def reshaping(self, attn_out):
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(self.batch_size, -1, self.hidden_dim)
        return attn_out


    def forward(self, query, key, value, mask):
        if self.self_attn:
            assert (query == key).all() and (key==value).all()
    
        self.batch_size = query.size(0)
        q = self.head_split(self.q_proj(query))
        k = self.head_split(self.k_proj(key))
        v = self.head_split(self.k_proj(value))

        attn_wts, attn_out = self.scaled_dot_product(q, k, v, mask)
        attn_out = self.attn_proj(self.reshaping(attn_out))

        return attn_wts, attn_out



# postion wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.FFN1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    
    def forward(self, x):
        output = self.FFN1(x)
        output = self.FFN2(output)
        return output



# a single encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, bias, dropout, layernorm_eps):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.self_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.dropout)


    def forward(self, x, mask):
        attn_wts, output = self.self_attention(query=x, key=x, value=x, mask=mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return attn_wts, output



# all encoders
class Encoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Encoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

        self.enc_num_layers = config.enc_num_layers
        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_head
        self.max_len = config.max_len
        self.bias = config.bias
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        self.pos_encoding = config.pos_encoding
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEncoding(self.max_len, self.hidden_dim, self.pos_encoding, self.device)
        self.encoders = nn.ModuleList([EncoderLayer(self.hidden_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.enc_num_layers)])


    def forward(self, x, mask=None):
        output = self.emb_layer(x) + self.pos_layer(x)
        output = self.dropout_layer(output)

        all_attn_wts = []
        for encoder in self.encoders:
            attn_wts, output = encoder(output, mask)
            all_attn_wts.append(attn_wts.detach().cpu())
        
        return all_attn_wts, output



# a single decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, bias, dropout, layernorm_eps):
        super(DecoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.masked_self_atention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=True)
        self.enc_dec_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=False, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.dropout)


    def forward(self, x, enc_output, dec_causal_mask, enc_dec_mask):
        dec_self_attn_wts, output = self.masked_self_atention(query=x, key=x, value=x, mask=dec_causal_mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        cross_attn_wts, output = self.enc_dec_attention(query=x, key=enc_output, value=enc_output, mask=enc_dec_mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return dec_self_attn_wts, cross_attn_wts, output



# all decoders
class Decoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Decoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

        self.dec_num_layers = config.dec_num_layers
        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_head
        self.max_len = config.max_len
        self.bias = config.bias
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        self.pos_encoding = config.pos_encoding

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEncoding(self.max_len, self.hidden_dim, self.pos_encoding, self.device)
        self.decoders = nn.ModuleList([DecoderLayer(self.hidden_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.dec_num_layers)])


    def forward(self, x, enc_output, dec_causal_mask=None, enc_dec_mask=None):
        output = self.emb_layer(x) + self.pos_layer(x)[:, :x.size(1)]
        output = self.dropout_layer(output)

        all_self_attn_wts, all_cross_attn_wts = [], []
        for decoder in self.decoders:
            dec_self_attn_wts, cross_attn_wts, output = decoder(output, enc_output, dec_causal_mask, enc_dec_mask)
            all_self_attn_wts.append(dec_self_attn_wts.detach().cpu())
            all_cross_attn_wts.append(cross_attn_wts.detach().cpu())
        
        return all_cross_attn_wts, output



# transformer
class Transformer(nn.Module):
    def __init__(self, config, tokenizers, device):
        super(Transformer, self).__init__()
        self.config = config
        self.src_tokenizer, self.trg_tokenizer = tokenizers
        self.device = device
        
        self.enc_num_layers = self.config.enc_num_layers
        self.dec_num_layers = self.config.dec_num_layers
        self.hidden_dim = self.config.hidden_dim
        self.num_head =  self.config.num_head
        self.max_len = self.config.max_len
        self.bias = bool(self.config.bias)
        self.dropout = self.config.dropout

        self.encoder = Encoder(self.config, self.src_tokenizer, self.device)
        self.decoder = Decoder(self.config, self.trg_tokenizer, self.device)
        self.fc = nn.Linear(self.hidden_dim, self.trg_tokenizer.vocab_size)
        
        self.init_weights()


    def init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)


    def make_mask(self, src, trg):
        enc_mask = torch.where(src==self.src_tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        dec_causal_mask = torch.tril(torch.ones(trg.size(1), trg.size(1))).unsqueeze(0).unsqueeze(1).to(self.device) + torch.where(trg==self.trg_tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        dec_causal_mask = torch.where(dec_causal_mask < 2, 0, 1)
        enc_dec_mask = enc_mask
        return enc_mask, dec_causal_mask, enc_dec_mask


    def forward(self, src, trg):
        enc_mask, dec_causal_mask, enc_dec_mask = self.make_mask(src, trg)
        all_attn_wts, enc_output = self.encoder(src, enc_mask)
        all_cross_attn_wts, output = self.decoder(trg, enc_output, dec_causal_mask, enc_dec_mask)
        output = self.fc(output)
        return all_cross_attn_wts, output