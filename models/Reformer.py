import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ReformerLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(
                        None, configs.d_model, configs.n_heads, bucket_size=configs.bucket_size, n_hashes=configs.n_hashes
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
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.enc_embedding(x_enc, None)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out)  # [B, L, d_model]
        enc_out = self.projection(enc_out)

        return enc_out
