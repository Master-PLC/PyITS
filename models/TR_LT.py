from math import sqrt

import numpy as np
import tednet.tnn.tensor_ring as tr
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import factorint

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from utils.masking import TriangularCausalMask


class AttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, r, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention

        input_size = factorint(seq_len, multiple=True) + [d_model]
        output_size = [seq_len, d_keys * n_heads]
        ranks = [r] * (len(input_size) + len(output_size))
        self.query_projection = tr.TRLinear(input_size, output_size, ranks)
        self.key_projection = tr.TRLinear(input_size, output_size, ranks)

        output_size = [seq_len, d_values * n_heads]
        self.value_projection = tr.TRLinear(input_size, output_size, ranks)
        self.out_projection = tr.TRLinear(output_size, input_size, ranks)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, **kwargs):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape(B, L, -1)
        keys = self.key_projection(keys).reshape(B, S, -1)
        values = self.value_projection(values).reshape(B, S, -1)

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out).reshape(B, L, -1)

        return out, attn


class FFNLayer(nn.Module):
    def __init__(self, seq_len, r, d_model, d_ff=None, dropout=0.1, activation="relu") -> None:
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model
        input_size = factorint(seq_len, multiple=True) + [d_model]
        output_size = [seq_len, d_ff]
        ranks = [r] * (len(input_size) + len(output_size))
        self.conv1 = tr.TRLinear(input_size, output_size, ranks)
        self.conv2 = tr.TRLinear(output_size, input_size, ranks)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, **kwargs):
        output = x
        output = self.conv1(output)
        output = self.dropout(self.activation(output))
        output = self.conv2(output)
        output = self.dropout(output)
        output = output.view(x.shape)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, seq_len, r, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFNLayer(seq_len, r, d_model, d_ff, dropout, activation)

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta, **kwargs
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.ffn(y, **kwargs)

        return self.norm2(x + y), attn


class LinearAttention(nn.Module):
    def __init__(self, seq_len, k, d_k, n_heads, r, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(LinearAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

        input_size = [n_heads, d_k, seq_len]
        output_size = [n_heads, d_k, k]
        ranks = [r] * (len(input_size) + len(output_size))
        self.proj_k = tr.TRLinear(input_size, output_size, ranks)
        self.proj_v = tr.TRLinear(input_size, output_size, ranks)
        if self.mask_flag:
            input_size = [1, seq_len, seq_len]
            output_size = [1, seq_len, k]
            ranks = [r] * (len(input_size) + len(output_size))
            self.proj_a = tr.TRLinear(input_size, output_size, ranks)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        keys = keys.permute(0, 2, 3, 1)  # [B, H, D, S]
        values = values.permute(0, 2, 3, 1)  # [B, H, D, S]

        keys = self.proj_k(keys)  # [B, H, D, k]
        values = self.proj_v(values)  # [B, H, D, k]

        keys = keys.reshape(B, H, D, -1)
        values = values.reshape(B, H, D, -1)

        keys = keys.permute(0, 1, 3, 2)  # [B, H, k, D]
        values = values.permute(0, 1, 3, 2)  # [B, H, k, D]
        queries = queries.permute(0, 2, 1, 3)  # [B, H, L, D]

        scores = torch.einsum("bhld,bhkd->bhlk", queries, keys)  # [B, H, L, k]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            attn_mask.mask = self.proj_k(attn_mask.mask).reshape(B, 1, L, -1)  # [B, 1, L, k]
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhlk,bhkd->bhld", A, values)  # [B, H, L, D]
        V = V.permute(0, 2, 1, 3)  # [B, L, H, D]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class Model(nn.Module):
    """A T^2-Tensor-Aided Multiscale Transformer for Remaining Useful Life Prediction in IIoT
    Paper link: https://ieeexplore.ieee.org/document/9756042
    """
    supported_tasks = ['rul_estimation']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.seq_len, configs.rank,
                    AttentionLayer(
                        LinearAttention(
                            configs.seq_len, configs.d_lower, 
                            configs.d_model // configs.n_heads, 
                            configs.n_heads, configs.rank,
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ),
                        configs.seq_len, configs.rank,
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=self.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None, **kwargs)  # [B, L, d_model]

        dec_out = self.projection(enc_out)
        return dec_out
