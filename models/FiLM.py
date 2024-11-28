import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy import special as ss

from layers.Decoders import OutputBlock


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A))  # [N, N]
        self.register_buffer('B', torch.Tensor(B))  # [N]
        vals = np.arange(0.0, 1.0, dt)  # [scale * P]
        self.register_buffer('eval_matrix', torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T))  # [scale * P, N]

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        # inputs: [B, Dx, L]
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(inputs.device)  # [B, Dx, N]
        cs = []
        for f in inputs.permute([-1, 0, 1]):  # [L, B, Dx]
            f = f.unsqueeze(-1)  # [B, Dx, 1]
            new = f @ self.B.unsqueeze(0)  # [B, Dx, N]
            c = F.linear(c, self.A) + new  # [B, Dx, N]
            cs.append(c)
        return torch.stack(cs, dim=0)  # [L, B, Dx, N]

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(
            torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
            torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real)
        )

    def forward(self, x):
        # x: [B, Dx, W, L]
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)  # B, Dx, W, L//2+1
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)  # B, Dx, W, L//2+1
        a = x_ft[:, :, :, :self.modes]  # B, Dx, W, L//2
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # [B, Dx, W, L]
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2205.08897
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.seq_len

        self.seq_len_all = self.seq_len + self.label_len

        self.layers = configs.e_layers
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        # b, s, f means b, f
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        self.multiscale = [1, 2, 4]
        self.window_size = [256]
        configs.ratio = 0.5
        self.legts = nn.ModuleList([
            HiPPO_LegT(N=n, dt=1. / self.pred_len / i) for n in self.window_size for i in self.multiscale
        ])
        self.spec_conv_1 = nn.ModuleList([
            SpectralConv1d(
                in_channels=n, out_channels=n, seq_len=min(self.pred_len, self.seq_len),
                ratio=configs.ratio
            ) for n in self.window_size for _ in range(len(self.multiscale))
        ])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        # Decoder
        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len  # scale * P
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2))  # [L, B, Dx, W]
            x_in_c = x_in_c.permute([1, 2, 3, 0])[:, :, :, jump_dist:]  # [B, Dx, W, L]
            out1 = self.spec_conv_1[i](x_in_c)  # [B, Dx, W, L]
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]  # [B, Dx, W]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T  # [B, Dx, P]
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)  # [B, Dx, P, n_scale * n_window]
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)  # [B, P, Dx]

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means  # [B, P, Dx]

        dec_out = self.projection(x_dec)
        return dec_out
