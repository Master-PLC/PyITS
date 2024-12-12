import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, DSAttention
from layers.Transformer_EncDec import (Decoder, DecoderLayer, Encoder,
                                       EncoderLayer)


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, 
            padding_mode='circular', bias=False
        )

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x: [B, L, D], stats: [B, 1, D]
        batch_size = x.shape[0]
        x = self.series_conv(x)  # [B, 1, D]
        x = torch.cat([x, stats], dim=1)  # [B, 2, D]
        x = x.view(batch_size, -1)  # [B, 2 * D]
        y = self.backbone(x)  # [B, O]

        return y


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ), configs.d_model, configs.n_heads
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
                            DSAttention(
                                True, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        AttentionLayer(
                            DSAttention(
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

        self.tau_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, 
            hidden_layers=configs.p_hidden_layers, output_dim=1
        )
        self.delta_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, 
            hidden_layers=configs.p_hidden_layers, output_dim=configs.seq_len
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # [B, 1, Dx]
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # [B, 1, Dx]
        x_enc = x_enc / std_enc

        # positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()  # [B, L, D] + [B, 1, D] -> [B, 1]
        delta = self.delta_learner(x_raw, mean_enc)  # [B, L, D] + [B, 1, D] -> [B, L]
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)  # [B, L, d_model]

        # Output
        if self.task_name in ['process_monitoring']:
            x_dec_new = torch.cat(
                [x_enc[:, -self.label_len:, -self.c_out:], torch.zeros_like(x_dec[:, -1:, :])], dim=1
            ).to(x_enc.device).clone()  # [B, S+1, Dy]

            dec_out = self.dec_embedding(x_dec_new, x_mark_dec)  # [B, S+1, d_model]
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)  # [B, S+1, Dy]
            dec_out = dec_out * std_enc[..., -self.c_out:] + mean_enc[..., -self.c_out:]
            dec_out = dec_out[:, -1:, :]
        else:
            dec_out = self.projection(enc_out)

        return dec_out
