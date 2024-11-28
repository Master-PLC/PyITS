import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from utils.tools import _ParameterDict


class DLSTMLayer(nn.Module):
    def __init__(self, enc_in, d_model):
        super().__init__()
        self.hidden = d_model
        self.LSTM = _ParameterDict({
            "W": nn.Parameter(torch.randn(enc_in, d_model * 4)),
            "U": nn.Parameter(torch.randn(d_model, d_model * 4)),
            "b": nn.Parameter(torch.zeros(d_model * 4)),
            "Wd": nn.Parameter(torch.randn(enc_in, d_model))
        })

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0, std=1. / np.sqrt(d_model))

    def forward(self, x):
        B, T, _ = x.size()

        hidden_seq = []

        h_t = torch.zeros(B, self.hidden).to(x.device)
        c_t = torch.zeros(B, self.hidden).to(x.device)

        for t in range(T):
            x_t = x[:, t, :]
            x_d = x[:, t, :] - x[:, t-1, :] if t > 0 else torch.zeros_like(x_t)

            gates = x_t @ self.LSTM["W"] + h_t @ self.LSTM["U"] + self.LSTM["b"]

            i_t = torch.sigmoid(gates[:, :self.hidden])
            f_t = torch.sigmoid(gates[:, self.hidden:self.hidden * 2])
            g_t = torch.tanh(gates[:, self.hidden * 2:self.hidden * 3])
            o_t = torch.sigmoid(gates[:, self.hidden * 3:] + x_d @ self.LSTM["Wd"])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


class Model(nn.Module):
    """DLSTM: A Novel Soft Sensor Modeling Approach Based on Difference-LSTM for Complex Industrial Process
    Paper link: https://ieeexplore.ieee.org/document/9531471
    """
    supported_tasks = ['soft_sensor']

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = configs.task_name

        self.DLSTM = nn.ModuleList([
            DLSTMLayer(
                configs.enc_in if i == 0 else configs.d_model, configs.d_model
            ) for i in range(configs.e_layers)
        ])

        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc shape: [B, L, D]
        output = x_enc
        ht, ct = [], []
        for dlstm in self.DLSTM:
            output, (h, c) = dlstm(output)
            ht.append(h)
            ct.append(c)

        dec_out = self.projection(output)
        return dec_out
