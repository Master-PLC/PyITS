import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelationLayer
from layers.Autoformer_EncDec import (Decoder, DecoderLayer, Encoder,
                                      EncoderLayer, my_Layernorm,
                                      series_decomp)
from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import (MultiWaveletCross,
                                            MultiWaveletTransform)


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = 1 if self.task_name in ['process_monitoring'] else configs.seq_len
        self.c_out = configs.c_out

        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model, out_channels=configs.d_model, 
                seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len, 
                modes=self.modes, ich=configs.d_model, base='legendre', activation='tanh'
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len,
                modes=self.modes, mode_select_method=self.mode_select
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model, 
                seq_len=self.seq_len // 2 + self.pred_len, modes=self.modes, 
                mode_select_method=self.mode_select
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len,
                modes=self.modes, mode_select_method=self.mode_select, num_heads=configs.n_heads
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            decoder_self_att, configs.d_model, configs.n_heads
                        ),
                        AutoCorrelationLayer(
                            decoder_cross_att, configs.d_model, configs.n_heads
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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None, **kwargs)  # [B, L, d_model]

        if self.task_name in ['process_monitoring']:
            # decomp init
            mean = torch.mean(x_enc, dim=1).unsqueeze(1)  # [B, 1, Dx]
            seasonal_init, trend_init = self.decomp(x_enc)  # [B, L, Dx], [B, L, Dx]
            # decoder input
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # [B, S+1, Dx]
            seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, 1))  # [B, S+1, Dx]

            # dec
            dec_out = self.dec_embedding(seasonal_init[..., -self.c_out:], x_mark_dec)  # [B, S+1, d_model]
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init[..., -self.c_out:])  # [B, S, Dy], [B, S, Dy]
            # final
            dec_out = trend_part + seasonal_part
            return dec_out[:, -1:, :]

        else:
            dec_out = self.projection(enc_out)
            return dec_out
