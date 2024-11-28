import torch
import torch.nn as nn

from layers.Decoders import OutputBlock


class Model(nn.Module):
    """FITS: Frequency Interpolation Time Series Forecasting
    paper link: https://arxiv.org/html/2307.03756v3
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)
                )

        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio)
            ).to(torch.cfloat)  # complex layer for frequency upcampling]

        # Decoder
        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=self.seq_len+self.pred_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x, *args, **kwargs):
        # x: [B, L, Dx]

        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, Dx]
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # [B, 1, Dx]
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)  # [B, L//2+1, Dx]
        # Low pass filter
        low_specx[:, self.dominance_freq:] = 0
        low_specx = low_specx[:, :self.dominance_freq, :]  # [B, Lp, Dx]

        if self.individual:
            low_specxy_ = torch.zeros([
                low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)  # [B, Lp*length_ratio, Dx]
            ], dtype=low_specx.dtype).to(low_specx.device)

            for i in range(self.channels):
                mlp = self.freq_upsampler[i]
                temp = low_specx[:, :, i]  # [B, Lp]
                low_specxy_[:, :, i] = mlp(temp)  # [B, Lp*length_ratio]
        else:
            low_specxy_ = self.freq_upsampler(
                low_specx.permute(0, 2, 1)  # [B, Dx, Lp]
            ).permute(0, 2, 1)  # [B, Lp*length_ratio, Dx]

        low_specxy = torch.zeros([
            low_specxy_.size(0), int((self.seq_len+self.pred_len)/2+1), low_specxy_.size(2)  # [B, (L+P)//2+1, Dx]
        ], dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, :low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)  # [B, L+P, Dx]
        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change

        xy = low_xy * torch.sqrt(x_var) + x_mean  # [B, L+P, Dx]
        y = self.projection(xy)
        return y
