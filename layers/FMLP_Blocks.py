import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterMLPBlock(nn.Module):
    """
    Canonical FMLP-Rec
    """
    def __init__(self, d_model, dropout, seq_len=16, init_ratio=1):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, seq_len//2 + 1, d_model, 2, dtype=torch.float32) * init_ratio)
        self.out_dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        weight = torch.view_as_complex(self.complex_weight)

        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FFNBlock(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)  # b x din x lq
        output = self.dropout(F.relu(self.w_1(output)))
        output = self.dropout(self.w_2(output).transpose(1, 2))
        output = self.layer_norm(output + residual)
        return output
