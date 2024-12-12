import torch.nn as nn

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import (Decoder, DecoderLayer, Encoder,
                                       EncoderLayer)


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://dl.acm.org/doi/pdf/10.5555/3295222.3295349
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ), 
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
        # Decoder
        if self.task_name in ['process_monitoring']:
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        AttentionLayer(
                            FullAttention(
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None, **kwargs)  # [B, L, d_model]

        if self.task_name in ['process_monitoring']:
            dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, S+P, Dy]
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, **kwargs)  # [B, S+P, Dy]
            return dec_out[:, -1:]

        else:
            dec_out = self.projection(enc_out)
            return dec_out
