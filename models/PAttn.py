import torch
import torch.nn as nn
from einops import rearrange

from layers.Decoders import OutputBlock
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=DV15UbHCY1
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride

        self.d_model = configs.d_model

        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                        ), configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(1)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.pred_len)
        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, Dx]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, Dx]
        x_enc = x_enc / stdev

        B, _, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)  # [B, Dx, L]
        x_enc = self.padding_patch_layer(x_enc)  # [B, Dx, L+stride]
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [B, Dx, patch_num, patch_size]

        enc_out = self.in_layer(x_enc)  # [B, Dx, patch_num, d_model]
        enc_out =  rearrange(enc_out, 'b c m l -> (b c) m l')  # [B*Dx, patch_num, d_model]
        dec_out, _ = self.encoder(enc_out)  # [B*Dx, patch_num, d_model]
        dec_out =  rearrange(dec_out, '(b c) m l -> b c (m l)' , b=B , c=C)  # [B, Dx, patch_num*d_model]
        dec_out = self.out_layer(dec_out)  # [B, Dx, P]
        dec_out = dec_out.permute(0, 2, 1)  # [B, P, Dx]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # [B, P, Dx]
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # [B, P, Dx]

        dec_out = self.projection(dec_out)
        return dec_out
