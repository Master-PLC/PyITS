from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.Crossformer_EncDec import (Decoder, DecoderLayer, Encoder,
                                       scale_block)
from layers.Decoders import OutputBlock
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import (AttentionLayer, FullAttention,
                                         TwoStageAttentionLayer)
from models.PatchTST import FlattenHead


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0
        )
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    configs, 1 if l == 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                    1, configs.dropout, self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l),
                    configs.factor
                ) for l in range(configs.e_layers)
            ]
        )

        # Decoder
        if self.task_name in ['process_monitoring']:
            self.dec_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

            self.decoder = Decoder(
                [
                    DecoderLayer(
                        TwoStageAttentionLayer(
                            configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, 
                            configs.n_heads, configs.d_ff, configs.dropout
                        ),
                        AttentionLayer(
                            FullAttention(
                                False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        self.seg_len,
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        # activation=configs.activation,
                    )
                    for l in range(configs.e_layers + 1)
                ],
            )
        
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.projection = OutputBlock(
                configs.enc_in, configs.c_out, seq_len=self.head_nf, pred_len=configs.pred_len, 
                task_name=self.task_name, dropout=configs.dropout
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)  # [B, Dx, in_seg_num, d_model]
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)  # (1 + e_layers) * [B, Dx, in_seg_num, d_model]

        if self.task_name in ['process_monitoring']:
            dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])  # [B, Dx, out_seg_num, d_model]
            dec_out = self.decoder(dec_in, enc_out)
            return dec_out[:, -self.pred_len:, -self.c_out:]

        else:
            enc_out = enc_out[-1].permute(0, 1, 3, 2)  # [B, Dx, d_model, out_seg_num]
            enc_out = self.flatten(enc_out).permute(0, 2, 1)  # [B, head_nf, Dx]
            dec_out = self.projection(enc_out)
            return dec_out
