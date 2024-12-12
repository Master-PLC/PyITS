import torch
import torch.nn as nn

from layers.Decoders import OutputBlock


class MultiplicationFilteringKernel(nn.Module):
    def __init__(self, kernel_type='WFK'):
        super(MultiplicationFilteringKernel, self).__init__()
        self.kernel_type = kernel_type

        self.center_freq = nn.Parameter(torch.rand(1, dtype=torch.float32))
        self.bandwidth = nn.Parameter(torch.rand(1, dtype=torch.float32))

    def forward(self, x):
        B, L, D = x.size()
        freqs = torch.arange(0, L, dtype=torch.float32, device=x.device)

        center_freq_clamped = torch.clamp(self.center_freq, 0, L-1)
        bandwidth_clamped = torch.clamp(self.bandwidth, min=0.1)  # 避免带宽为0

        if self.kernel_type == 'WFK':
            kernel = 1 / (1 + 2 * bandwidth_clamped * (freqs - center_freq_clamped) ** 2)
        elif self.kernel_type == 'GFK':
            kernel = torch.exp(-(freqs - center_freq_clamped) ** 2 / (2 * bandwidth_clamped ** 2))
        else:
            raise ValueError("Unknown kernel type")
        kernel = kernel.view(1, L, 1)
        x = x * kernel
        return x


class Model(nn.Module):
    """MCN: An Interpretable Multiplication-Convolution Network for Equipment Intelligent Edge Diagnosis
    Paper link: https://ieeexplore.ieee.org/document/10443049?denied=
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.kernels = nn.ModuleList([
            MultiplicationFilteringKernel(kernel_type=configs.kernel_type) for _ in range(configs.n_kernels)
        ])
        self.conv = nn.Conv2d(configs.enc_in, configs.d_model, kernel_size=7, padding=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=1, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x shape: [B, L, D]
        xf = torch.fft.rfft(x_enc, dim=1)  # [B, L//2+1, D]
        xf = xf.abs()

        h_temp = [kernel(xf) for kernel in self.kernels]
        h = [kernel(xf - sum(h_temp[:i])) for i, kernel in enumerate(self.kernels)]
        h = torch.stack(h, dim=1)  # [B, N, L//2+1, D]
        h = h.permute(0, 3, 1, 2)  # [B, D, N, L//2+1]

        h = self.conv(h)  # [B, d_model, N, L//2+1]
        h = self.global_pool(h).squeeze(-1)  # [B, d_model, 1]
        h = h.transpose(1, 2)  # [B, 1, d_model]
        out = self.projection(h)
        return out
