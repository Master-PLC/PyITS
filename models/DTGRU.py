import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from utils.tools import _ParameterDict


class DTGRULayer(nn.Module):
    def __init__(self, enc_in, d_model):
        super().__init__()
        self.hidden = d_model
        self.GRU = _ParameterDict({
            "Wa": nn.Parameter(torch.randn(d_model, d_model * 2)),  # 更新门、重置门和候选状态
            "Ua": nn.Parameter(torch.randn(d_model, d_model * 2)),
            "ba": nn.Parameter(torch.zeros(d_model * 2)),
            "Wna": nn.Parameter(torch.randn(d_model, d_model)),  # 更新门、重置门和候选状态
            "Una": nn.Parameter(torch.randn(d_model, d_model)),
            "bna": nn.Parameter(torch.zeros(d_model)),

            "WAttn": nn.Parameter(torch.randn(d_model, d_model)),  # 更新门、重置门和候选状态
            "UAttn": nn.Parameter(torch.randn(d_model, d_model)),
            "VAttn": nn.Parameter(torch.randn(d_model, d_model)),

            "Wma": nn.Parameter(torch.randn(d_model, d_model)),  # 更新门、重置门和候选状态
            "Uma": nn.Parameter(torch.randn(d_model, d_model)),
            "bma": nn.Parameter(torch.randn(d_model)),

            "Wm": nn.Parameter(torch.randn(enc_in, d_model * 2)),  # 更新门、重置门和候选状态
            "Um": nn.Parameter(torch.randn(d_model, d_model * 2)),
            "bm": nn.Parameter(torch.zeros(d_model * 2)),
            "Whm": nn.Parameter(torch.randn(enc_in, d_model)),  # 更新门、重置门和候选状态
            "Uhm": nn.Parameter(torch.randn(d_model, d_model)),
            "bhm": nn.Parameter(torch.zeros(d_model)),
        })

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.normal_(param, mean=0, std=1. / np.sqrt(d_model))

    def forward(self, x):
        B, T, _ = x.size()

        hidden_seq = []
        h_t = torch.zeros(B, self.hidden).to(x.device)
        N_t = torch.zeros(B, self.hidden).to(x.device)

        for t in range(T):
            x_t = x[:, t, :]

            if t == 0:
                d_t = torch.zeros_like(h_t)
            elif t == 1:
                d_t = hidden_seq[-1]
            else:
                d_t = hidden_seq[-1] - hidden_seq[-2]

            gates_A = d_t @ self.GRU["Wa"] + N_t @ self.GRU["Ua"] + self.GRU["ba"]  # [B, 2*hidden]
            z_t_A = torch.sigmoid(gates_A[:, :self.hidden])  # [B, hidden]
            r_t_A = torch.sigmoid(gates_A[:, self.hidden:])
            hat_N_t = torch.tanh(d_t @ self.GRU["Wna"] + (N_t * r_t_A) @ self.GRU["Una"] + self.GRU["bna"])
            N_t = (1 - z_t_A) * N_t + z_t_A * hat_N_t

            score = torch.tanh(h_t @ self.GRU["WAttn"] + N_t @ self.GRU["UAttn"])
            score = score @ self.GRU["VAttn"]
            alpha = F.softmax(score, dim=1)
            Omega_t = alpha * N_t
            m_t = torch.tanh(Omega_t @ self.GRU["Wma"] + h_t @ self.GRU["Uma"] + self.GRU["bma"])

            gates_M = x_t @ self.GRU["Wm"] + m_t @ self.GRU["Um"] + self.GRU["bm"]
            z_t_M = torch.sigmoid(gates_M[:, :self.hidden])
            r_t_M = torch.sigmoid(gates_M[:, self.hidden:])
            hat_h_t = torch.tanh(x_t @ self.GRU["Whm"] + (m_t * r_t_M) @ self.GRU["Uhm"] + self.GRU["bhm"])
            h_t = (1 - z_t_M) * h_t + z_t_M * hat_h_t

            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq, (h_t, N_t)


class Model(nn.Module):
    """DTGRU: Dual-Thread Gated Recurrent Unit for Gear  Remaining Useful Life Prediction
    Paper link: https://ieeexplore.ieee.org/document/9931971
    """
    supported_tasks = ['rul_estimation']

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = configs.task_name

        self.DTGRU = nn.ModuleList([
            DTGRULayer(
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
        ht, Nt = [], []
        for dtgru in self.DTGRU:
            output, (h, N) = dtgru(output)
            ht.append(h)
            Nt.append(N)

        dec_out = self.projection(output)
        return dec_out
