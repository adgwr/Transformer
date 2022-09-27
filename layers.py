import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_padding_mask):
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, src_padding_mask)
            torch.cuda.empty_cache()
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_head, dim_qk=d_model, dim_v=d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_padding_mask):
        # Multi-Head Attention
        src2 = self.self_attention(src, src, src, src_padding_mask)
        # Add and Norm
        src = self.norm1(src + self.dropout1(src2))
        # Feed Forward
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        # Add and Norm
        src = self.norm2(src + self.dropout3(src2))
        return src


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, src_padding_mask, tgt_padding_mask, tgt_subsequent_mask):
        output = tgt
        for i in range(self.num_layers):
            output = self.layers[i](output, memory, src_padding_mask, tgt_padding_mask, tgt_subsequent_mask)
            torch.cuda.empty_cache()
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dim_qk=d_model, dim_v=d_model, dropout=dropout)
        self.enc_attn = MultiheadAttention(d_model, n_head, dim_qk=d_model, dim_v=d_model, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_padding_mask, tgt_padding_mask, tgt_subsequent_mask):
        # Masked Multi-Head Attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_padding_mask, tgt_subsequent_mask)
        # Add and Norm
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        # Multi-Head Attention
        tgt2 = self.enc_attn(tgt, memory, memory, src_padding_mask)
        # Add and Norm
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        # Feed Forward
        tgt2 = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        # Add and Norm
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dim_qk, dim_v, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.dim_qk_per_head = dim_qk // num_heads
        self.dim_v_per_head = dim_v // num_heads
        assert self.dim_qk_per_head * num_heads == dim_qk, 'dim_qk must be divisible by num_heads'
        assert self.dim_v_per_head * num_heads == dim_v, 'dim_v must be divisible by num_heads'
        self.qkv_same_d_model = (dim_qk == d_model and dim_v == d_model)
        if self.qkv_same_d_model:
            self.w = nn.Parameter(torch.Tensor(3 * d_model, d_model))
        else:
            self.w_q = nn.Parameter(torch.Tensor(dim_qk, d_model))
            self.w_k = nn.Parameter(torch.Tensor(dim_qk, d_model))
            self.w_v = nn.Parameter(torch.Tensor(dim_v, d_model))
        self.fc = nn.Linear(dim_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, padding_mask=None, subsequent_mask=None):
        '''
        Inputs:
        :param q: (len_q, batch_size, d_model)
        :param k: (len_kv, batch_size, d_model)
        :param v: (len_kv, batch_size, d_model)
        In Multi_Head Attention: len_q = len_kv = src_len or tgt_len
        In Decoder Encoder Attention: len_q = tgt_len, len_kv = src_len
        :param padding_mask: (batch_size, len_kv=src_len)
        :param subsequent_mask: (len_q=tgt_len, len_q)
        :return:
        '''
        len_q, batch_size, d_model = q.size()
        len_kv = k.size(0)
        # 计算q, k, v. 将维度相同的矩阵乘法合并，加快运行速度
        # q: (len_q, batch_size, d_model) -> (len_q, batch_size, dim_qk)
        # k: (len_kv, batch_size, d_model) -> (len_kv, batch_size, dim_qk)
        # v: (len_kv, batch_size, d_model) -> (len_kv, batch_size, dim_v)
        kv_same = torch.equal(k, v)
        qkv_same = kv_same and torch.equal(q, k)
        if self.qkv_same_d_model:
            # q, k ,v的维度均为d_model
            if qkv_same:
                q, k, v = F.linear(q, self.w).chunk(3, dim=-1)
            elif kv_same:
                q = F.linear(q, self.w[0:self.dim_qk, :])
                k, v = F.linear(k, self.w[self.dim_qk:, :]).chunk(2, dim=-1)
            else:
                q = F.linear(q, self.w[0:self.dim_qk, :])
                k_end = self.dim_qk * 2
                k = F.linear(k, self.w[self.dim_qk:k_end, :])
                v = F.linear(v, self.w[k_end:, :])
        else:
            # q, k, v的维度不同于d_model
            q = F.linear(q, self.w_q)
            k = F.linear(k, self.w_k)
            v = F.linear(v, self.w_v)

        # 拆分为多头
        bsz_n_heads = batch_size * self.num_heads
        # q: (batch_size * n_heads, len_q, dim_qk_per_head)
        q = q.contiguous().view(len_q, bsz_n_heads, self.dim_qk_per_head).transpose(0, 1)
        # k: (batch_size * n_heads, len_kv, dim_qk_per_head)
        k = k.contiguous().view(len_kv, bsz_n_heads, self.dim_qk_per_head).transpose(0, 1)
        # v: (batch_size * n_heads, len_kv, dim_v_per_head)
        v = v.contiguous().view(len_kv, bsz_n_heads, self.dim_v_per_head).transpose(0, 1)

        # 计算q * k
        # scores: (batch_size * n_heads, len_q, len_kv)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.dim_v_per_head)

        # mask
        if padding_mask is not None and subsequent_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand(batch_size, len_q, len_kv).unsqueeze(1)
            padding_mask = subsequent_mask + padding_mask
            scores = scores.view(batch_size, self.num_heads, len_q, len_kv)
            scores.masked_fill_(padding_mask, float('-inf'))
            scores = scores.view(bsz_n_heads, len_q, len_kv)
        else:
            if padding_mask is not None:
                scores = scores.view(batch_size, self.num_heads, len_q, len_kv)
                scores.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                scores = scores.view(bsz_n_heads, len_q, len_kv)
            if subsequent_mask is not None:
                scores.masked_fill_(subsequent_mask.unsqueeze(0), float('-inf'))

        scores = self.dropout(F.softmax(scores, dim=-1))
        # scores[scores != scores] = 0.0
        # (batch_size * n_heads, len_q, dim_v_per_head)
        attn_output = torch.matmul(scores, v)

        # 合并多头
        # (len_q, batch_size, dim_v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(len_q, batch_size, self.dim_v)
        # (len_q, batch_size, d_model)
        return self.fc(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False).to(x.device)
        return self.dropout(x)
