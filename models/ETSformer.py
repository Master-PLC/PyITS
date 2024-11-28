import torch
import torch.nn as nn

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.ETSformer_EncDec import (Decoder, DecoderLayer, Encoder,
                                     EncoderLayer, Transform)


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2202.01381
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len if self.task_name in ['process_monitoring'] else configs.seq_len
        self.c_out = configs.c_out

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in, configs.seq_len, 
                    self.pred_len, configs.top_k, dim_feedforward=configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )

        if self.task_name in ['process_monitoring']:
            # Decoder
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        configs.d_model, configs.n_heads, configs.c_out, self.pred_len,
                        dropout=configs.dropout,
                    ) for _ in range(configs.d_layers)
                ],
            )
            self.transform = Transform(sigma=0.2)

        else:
            self.projection = OutputBlock(
                configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
                task_name=self.task_name, dropout=configs.dropout
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['process_monitoring']:
            with torch.no_grad():
                if self.training:
                    x_enc = self.transform.transform(x_enc)  # [B, L, Dx]
        res = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)  # [B, L, Dx], e_layers * [B, L+1, d_model], e_layers * [B, L+P, d_model]

        if self.task_name in ['process_monitoring']:
            growth, season = self.decoder(growths, seasons)  # [B, P, c_out], [B, P, c_out]
            preds = level[:, -1:, -self.c_out:] + growth + season
            return preds
        else:
            growths = torch.sum(torch.stack(growths, 0), 0)[:, :self.seq_len, :]  # [B, L, d_model]
            seasons = torch.sum(torch.stack(seasons, 0), 0)[:, :self.seq_len, :]  # [B, L, d_model]

            preds = growths + seasons
            return self.projection(preds)
