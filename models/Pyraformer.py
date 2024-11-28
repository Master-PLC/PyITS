import torch
import torch.nn as nn

from layers.Decoders import OutputBlock
from layers.Pyraformer_EncDec import Encoder


class Model(nn.Module):
    """ 
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.seq_len
        self.d_model = configs.d_model

        self.encoder = Encoder(configs, configs.window_size, configs.inner_size)
        self.head = nn.Linear((len(configs.window_size)+1)*self.d_model, self.pred_len * configs.enc_in)
        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]  # [B, L, len(all_size)*d_model] -> [B, len(all_size)*d_model]
        dec_out = self.head(enc_out)  # [B, P*Dx]
        dec_out = dec_out.reshape(enc_out.size(0), self.pred_len, -1)  # [B, P, Dx]
        dec_out = self.projection(dec_out)
        return dec_out
