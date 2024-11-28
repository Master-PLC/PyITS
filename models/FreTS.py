import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.feature_size = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
        )

        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=self.hidden_size, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def tokenEmb(self, x):
        """dimension extension
        """
        # x: [B, L, Dx]
        x = x.permute(0, 2, 1)  # [B, Dx, L]
        x = x.unsqueeze(3)  # [B, Dx, L, 1]
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings  # [1, E]
        return x * y  # [B, Dx, L, E]

    def MLP_temporal(self, x, B, N, L):
        """frequency temporal learner
        """
        # x: [B, Dx, L, E]
        # FFT on L dimension
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, Dx, L//2+1, E]
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)  # [B, Dx, L//2+1, E]
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")  # [B, Dx, L, E]
        return x

    def MLP_channel(self, x, B, N, L):
        """frequency channel learner
        """
        # x: [B, Dx, L, E]
        x = x.permute(0, 2, 1, 3)  # [B, L, Dx, E]
        # FFT on Dx dimension
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, L, Dx//2+1, E]
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)  # [B, L, Dx//2+1, E]
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")  # [B, L, Dx, E]
        x = x.permute(0, 2, 1, 3)  # [B, Dx, L, E]
        return x

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        """frequency-domain MLPs
        Args:
            dimension: FFT along the dimension
            r: the real part of weights
            i: the imaginary part of weights
            rb: the real part of bias
            ib: the imaginary part of bias
        """
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)  # [B, L, Dx//2+1, E]
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)  # [B, L, Dx//2+1, E]

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb  # [E]
        )  # [B, L, Dx//2+1, E]

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib  # [E]
        )  # [B, L, Dx//2+1, E]

        y = torch.stack([o1_real, o1_imag], dim=-1)  # [B, L, Dx//2+1, E, 2]
        y = F.softshrink(y, lambd=self.sparsity_threshold)  # [B, L, Dx//2+1, E]
        y = torch.view_as_complex(y)
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x: [B, L, Dx]
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)  # [B, Dx, L, E]
        bias = x  # [B, Dx, L, E]
        if self.channel_independence == 0:
            x = self.MLP_channel(x, B, N, T)  # [B, Dx, L, E]
        x = self.MLP_temporal(x, B, N, T)  # [B, Dx, L, E]
        x = x + bias  # [B, Dx, L, E]
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, hidden_size, Dx]

        dec_out = self.projection(x)
        return dec_out
