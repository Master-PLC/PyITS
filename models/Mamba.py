import math

import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding


class Model(nn.Module):
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.seq_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )

        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)

        x_out = self.projection(x)
        return x_out
