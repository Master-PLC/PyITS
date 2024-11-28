from math import sqrt

import numpy as np

import torch
import torch.nn as nn
from einops import rearrange, repeat
from reformer_pytorch import LSHSelfAttention
from utils.masking import ProbMask, TriangularCausalMask


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # [B, Lq, H, Dh]
        _, S, _, D = values.shape  # [B, Lv, H, Dh]
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, Lv]

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta  # [B, H, Lq, Lv]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, Lq, Lv]
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [B, Lq, H, Dh]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class LinearAttention(nn.Module):
    def __init__(self, seq_len, k, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(LinearAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

        self.proj_k = nn.Linear(seq_len, k, bias=False)
        self.proj_v = nn.Linear(seq_len, k, bias=False)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        keys = keys.permute(0, 2, 3, 1)  # [B, H, D, S]
        values = values.permute(0, 2, 3, 1)  # [B, H, D, S]

        keys = self.proj_k(keys)  # [B, H, D, k]
        values = self.proj_v(values)  # [B, H, D, k]

        keys = keys.permute(0, 1, 3, 2)  # [B, H, k, D]
        values = values.permute(0, 1, 3, 2)  # [B, H, k, D]
        queries = queries.permute(0, 2, 1, 3)  # [B, H, L, D]

        scores = torch.einsum("bhld,bhkd->bhlk", queries, keys)  # [B, H, L, k]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            attn_mask.mask = self.proj_k(attn_mask.mask)  # [B, 1, L, k]
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhlk,bhkd->bhld", A, values)  # [B, H, L, D]
        V = V.permute(0, 2, 1, 3)  # [B, L, H, D]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Args:
            n_top: c * ln(L_q)
        """
        # Q: [B, H, Lq, D], K: [B, H, Lk, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # [B, H, Lq, Lk, Dh]
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # [Lq, sample_k]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # [B, H, Lq, sample_k, Dh]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2),  # [B, H, Lq, 1, Dh]
            K_sample.transpose(-2, -1)  # [B, H, Lq, Dh, sample_k]
        ).squeeze()  # [B, H, Lq, sample_k]

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # [B, H, Lq]
        M_top = M.topk(n_top, sorted=False)[1]  # [B, H, n_top]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top,  # factor * ln(L_q)
            :
        ]  # [B, H, n_top, Dh]
        Q_K = torch.matmul(
            Q_reduce,  # [B, H, n_top, Dh]
            K.transpose(-2, -1)  # [B, H, Dh, Lk]
        )  # [B, H, n_top, Lk]

        return Q_K, M_top  # [B, H, n_top, Lk], [B, H, n_top]

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  # [B, H, Lv, Dh]
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)  # [B, H, Dh]
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # [B, H, Lq, Dh]

        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)  # [B, H, Lq, Dh]
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape  # [B, H, Lv, Dh]

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # [B, H, n_top, Lk]

        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)  # [B, H, n_top, Dh]
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # [B, H, Lq, Dh]
        keys = keys.transpose(2, 1)  # [B, H, Lk, Dh]
        values = values.transpose(2, 1)  # [B, H, Lv, Dh]

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c * ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c * ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)  # [B, H, n_top, Lk], [B, H, n_top]

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # [B, H, n_top, Lk]
        # get the context
        context = self._get_initial_context(values, L_Q)  # [B, H, Lq, Dh]
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)  # [B, H, Lq, Dh]

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, **kwargs):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

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

        return self.out_projection(out), attn


class LocalAttentionLayer(nn.Module):
    def __init__(self, attention, seq_len, d_model, kernel_size, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)

        self.out_len = int(np.ceil(seq_len / kernel_size))
        padding = self.out_len * kernel_size - seq_len if seq_len % kernel_size != 0 else 0
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=kernel_size, stride=kernel_size, padding=padding)
        self.value_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=kernel_size, stride=kernel_size, padding=padding)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, **kwargs):
        B, L, _ = queries.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys.transpose(1, 2)).transpose(1, 2)
        values = self.value_projection(values.transpose(1, 2)).transpose(1, 2)

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, self.out_len, H, -1)
        values = values.view(B, self.out_len, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(
        self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, 
        n_hashes=4
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model, heads=n_heads, bucket_size=bucket_size, n_hashes=n_hashes, causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape  # [B, L, D]
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))  # [B, L, D]
        queries = queries[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            FullAttention(
                False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention
            ), d_model, n_heads
        )
        self.dim_sender = AttentionLayer(
            FullAttention(
                False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention
            ), d_model, n_heads
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(
                False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention
            ), d_model, n_heads
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
