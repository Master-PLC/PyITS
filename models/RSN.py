import numpy as np
import torch
import torch.nn as nn

from layers.Decoders import OutputBlock


class Model(nn.Module):
    """RSN: Restricted Sparse Networks for Rolling Bearing Fault Diagnosis
    Paper link: https://ieeexplore.ieee.org/document/10043748/
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.kernel_size = 16
        self.padding = 8
        self.stride = 2
        self.kernel_size_pool = 2

        self.freq_len = self.seq_len // 2 + 1
        self.output_len_conv1d = int(np.floor((self.freq_len + 2 * self.padding - self.kernel_size) / self.stride)) + 1
        self.output_len_pooling = int(np.floor((self.output_len_conv1d - self.kernel_size_pool) / self.stride)) + 1

        self.real_filter = nn.Sequential(
            nn.Conv1d(self.enc_in, self.d_model, self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Tanh(),
            nn.MaxPool1d(self.kernel_size_pool, stride=self.stride)
        )
        self.imag_filter = nn.Sequential(
            nn.Conv1d(self.enc_in, self.d_model, self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Tanh(),
            nn.MaxPool1d(self.kernel_size_pool, stride=self.stride)
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Tanh()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Tanh()
        )
        self.layernorm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.output_len_pooling * 2, self.seq_len),
            nn.ReLU()
        )

        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x shape: [B, L, D]
        x = torch.fft.rfft(x_enc, dim=1)
        x_real, x_imag = x.real, x.imag  # [B, L//2+1, D]

        x_real = self.real_filter(x_real.transpose(1, 2))  # [B, d_model, L//8]
        x_imag = self.imag_filter(x_imag.transpose(1, 2))  # [B, d_model, L//8]

        x_comb = torch.cat([x_real, x_imag], dim=1).transpose(1, 2)  # [B, L//8, 2*d_model]
        x_real = self.mlp1(x_comb).transpose(1, 2)  # [B, d_model, L//8]
        x_imag = self.mlp2(x_comb).transpose(1, 2)  # [B, d_model, L//8]

        A = -2 * x_real  # [B, d_model, L//8]
        B = x_real ** 2 + x_imag ** 2  # [B, d_model, L//8]

        Z = A ** 2 - B
        Z = torch.softmax(Z, dim=1)  # [B, d_model, L//8]

        A = A * Z  # [B, d_model, L//8]
        B = B * Z  # [B, d_model, L//8]

        x_real = A * B + (A ** 2 - B) * x_real  # [B, d_model, L//8]
        x_imag = (A ** 2 - B) * x_imag  # [B, d_model, L//8]

        dec_in = torch.cat([x_real, x_imag], dim=-1).transpose(1, 2)  # [B, L//4, d_model]
        dec_in = self.layernorm(dec_in)
        dec_in = self.head(dec_in.transpose(1, 2)).transpose(1, 2)  # [B, L, d_model]

        dec_out = self.projection(dec_in)
        return dec_out
