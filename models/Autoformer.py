import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import (Decoder, DecoderLayer, Encoder,
                                      EncoderLayer, my_Layernorm,
                                      series_decomp)
from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.c_out = configs.c_out

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Decoder
        if self.task_name in ['process_monitoring']:
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                True, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        else:
            self.projection = OutputBlock(
                configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
                task_name=self.task_name, dropout=configs.dropout
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, **kwargs)

        if self.task_name in ['process_monitoring']:
            # decomp init
            mean = torch.mean(x_enc[..., -self.c_out:], dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, P, Dy]
            zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)  # [B, P, Dy]
            seasonal_init, trend_init = self.decomp(x_enc[..., -self.c_out:])

            # decoder input
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # [B, S+P, Dy]
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # [B, S+P, Dy]

            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)  # [B, S+P, d_model]
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)  # [B, S+P, Dy], [B, S+P, Dy]
            # final
            dec_out = trend_part + seasonal_part
            return dec_out[:, -self.pred_len:]
        else:
            dec_out = self.projection(enc_out)
            return dec_out
