import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock


class Model(nn.Module):
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = configs.task_name
        self.lstm = nn.LSTM(
            configs.enc_in, configs.d_model, configs.e_layers, dropout=configs.dropout, batch_first=True
        )
        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x shape: [B, L, D]
        x, _ = self.lstm(x_enc)
        x = self.projection(x)
        return x
