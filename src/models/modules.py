import torch
import torch.nn as nn
import torch.nn.functional as F



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
        attn_wts = torch.matmul(q, torch.transpose(k, 2, 3))/(self.head_dim ** 0.5)
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
        v = self.head_split(self.v_proj(value))

        attn_wts, attn_out = self.scaled_dot_product(q, k, v, mask)
        attn_out = self.attn_proj(self.reshaping(attn_out))

        return attn_wts, attn_out



# postion wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout, bias):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.bias = bias

        self.FFN1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim, bias=self.bias),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(self.ffn_dim, self.hidden_dim, bias=self.bias),
        )
        self.init_weights()


    def init_weights(self):
        for _, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.02)

    
    def forward(self, x):
        output = self.FFN1(x)
        output = self.FFN2(output)
        return output



# single encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.self_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.ffn_dim, self.dropout, self.bias)


    def forward(self, x, mask):
        attn_wts, output = self.self_attention(query=x, key=x, value=x, mask=mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return attn_wts, output
    

# single decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(DecoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.masked_self_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=True)
        self.enc_dec_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=False, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.ffn_dim, self.dropout, self.bias)


    def forward(self, x, enc_output, dec_causal_mask, enc_dec_mask):
        dec_self_attn_wts, output = self.masked_self_attention(query=x, key=x, value=x, mask=dec_causal_mask)
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