import torch
import torch.nn as nn

from models.modules import EncoderLayer, DecoderLayer
from models.embeddings import TokenEmbeddings, PositionalEmbedding



# transformer encoder
class Encoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Encoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

        self.enc_num_layers = config.enc_num_layers
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_head = config.num_heads
        self.max_len = config.max_len
        self.bias = bool(config.bias)
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        self.pos_encoding = config.pos_encoding
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = TokenEmbeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEmbedding(self.max_len, self.hidden_dim, self.pos_encoding, self.device)
        self.encoders = nn.ModuleList([EncoderLayer(self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.enc_num_layers)])


    def forward(self, x, mask=None):
        output = self.emb_layer(x) + self.pos_layer(x)
        output = self.dropout_layer(output)

        all_attn_wts = []
        for encoder in self.encoders:
            attn_wts, output = encoder(output, mask)
            all_attn_wts.append(attn_wts.detach().cpu())
        
        return all_attn_wts, output



# transformer decoders
class Decoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Decoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

        self.dec_num_layers = config.dec_num_layers
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_head = config.num_heads
        self.max_len = config.max_len
        self.bias = bool(config.bias)
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        self.pos_encoding = config.pos_encoding

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = TokenEmbeddings(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PositionalEmbedding(self.max_len, self.hidden_dim, self.pos_encoding, self.device)
        self.decoders = nn.ModuleList([DecoderLayer(self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.dec_num_layers)])


    def forward(self, x, enc_output, dec_causal_mask=None, enc_dec_mask=None):
        output = self.emb_layer(x) + self.pos_layer(x)
        output = self.dropout_layer(output)

        all_self_attn_wts, all_cross_attn_wts = [], []
        for decoder in self.decoders:
            dec_self_attn_wts, cross_attn_wts, output = decoder(output, enc_output, dec_causal_mask, enc_dec_mask)
            all_self_attn_wts.append(dec_self_attn_wts.detach().cpu())
            all_cross_attn_wts.append(cross_attn_wts.detach().cpu())
        
        return all_cross_attn_wts, output



# transformer
class Transformer(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Transformer, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        self.hidden_dim = self.config.hidden_dim

        self.encoder = Encoder(self.config, self.tokenizer, self.device)
        self.decoder = Decoder(self.config, self.tokenizer, self.device)
        self.fc = nn.Linear(self.hidden_dim, self.tokenizer.vocab_size)


    def make_mask(self, src, trg):
        enc_mask = torch.where(src==self.tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        dec_causal_mask = torch.tril(torch.ones(trg.size(1), trg.size(1))).unsqueeze(0).unsqueeze(1).to(self.device) + torch.where(trg==self.tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        dec_causal_mask = torch.where(dec_causal_mask < 2, 0, 1)
        enc_dec_mask = enc_mask
        return enc_mask, dec_causal_mask, enc_dec_mask


    def forward(self, src, trg):
        enc_mask, dec_causal_mask, enc_dec_mask = self.make_mask(src, trg)
        all_attn_wts, enc_output = self.encoder(src, enc_mask)
        all_cross_attn_wts, output = self.decoder(trg, enc_output, dec_causal_mask, enc_dec_mask)
        output = self.fc(output)
        return all_cross_attn_wts, output