import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.Transformer_EncDec import (ConvLayer, Decoder, DecoderLayer,
                                       Encoder, EncoderLayer)


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
            ] if configs.distil and self.task_name in ['process_monitoring'] else None,
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        if self.task_name in ['process_monitoring']:
            # Decoder
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            ProbAttention(
                                True, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        AttentionLayer(
                            ProbAttention(
                                False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        else:
            self.projection = OutputBlock(
                configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
                task_name=self.task_name, dropout=configs.dropout
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # [B, 1, Dx]
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # [B, 1, Dx]
        x_enc = x_enc / std_enc

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [B, L, d_model]

        if self.task_name in ['process_monitoring']:
            # x_dec: [B, S+P, Dy]
            dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, S+P, d_model]
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)  # [B, S+P, Dy]
            dec_out = dec_out[:, -self.pred_len:, :]
            dec_out = dec_out * std_enc[..., -self.c_out:] + mean_enc[..., -self.c_out:]

        else:
            dec_out = self.projection(enc_out)

        return dec_out
