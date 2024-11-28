import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock


class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        self._build()

    def _build(self):
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        # x: [B * Dx, C, Nc]
        x = self.spatial_proj(x.permute(0, 2, 1))  # [B * Dx, Nc, H//4]
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))  # [B * Dx, H//4, Nc]
        x = self.output_proj(x.permute(0, 2, 1))  # [B * Dx, Nc, O]
        x = x.permute(0, 2, 1)  # [B * Dx, O, Nc]

        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2207.01186
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.chunk_size = min(configs.seq_len, configs.chunk_size)
        # assert (self.seq_len % self.chunk_size == 0)
        if self.seq_len % self.chunk_size != 0:
            self.padding = nn.ConstantPad1d((self.chunk_size - self.seq_len % self.chunk_size, 0), 0)
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)  # padding in order to ensure complete division
        else:
            self.padding = None
        self.num_chunks = self.seq_len // self.chunk_size

        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.dropout = configs.dropout

        self._build()

        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def _build(self):
        self.ar = nn.Linear(self.seq_len, self.pred_len)

        self.layer_1 = IEBlock(
            input_dim=self.chunk_size, hid_dim=self.d_model // 4, 
            output_dim=self.d_model // 4, num_node=self.num_chunks
        )

        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size, hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4, num_node=self.num_chunks
        )

        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2, hid_dim=self.d_model // 2,
            output_dim=self.pred_len, num_node=self.enc_in
        )

    def encoder(self, x):
        B, T, N = x.size()

        if self.padding is not None:
            x = self.padding(x.permute(0, 2, 1)).permute(0, 2, 1)
        highway = self.ar(x.permute(0, 2, 1))  # [B, Dx, P]
        highway = highway.permute(0, 2, 1)  # [B, P, Dx]

        # continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)  # [B, Nc, C, Dx]
        x1 = x1.permute(0, 3, 2, 1)  # [B, Dx, C, Nc]
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)  # [B * Dx, C, Nc]
        x1 = self.layer_1(x1)  # [B * Dx, d_model//4, Nc]
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)  # [B * Dx, d_model//4]

        # interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)  # [B, C, Nc, Dx]
        x2 = x2.permute(0, 3, 1, 2)  # [B, Dx, C, Nc]
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)  # [B * Dx, C, Nc]
        x2 = self.layer_2(x2)  # [B * Dx, d_model//4, Nc]
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)  # [B * Dx, d_model//4]

        x3 = torch.cat([x1, x2], dim=-1)  # [B * Dx, d_model//2]
        x3 = x3.reshape(B, N, -1)  # [B, Dx, d_model//2]
        x3 = x3.permute(0, 2, 1)  # [B, d_model//2, Dx]

        out = self.layer_3(x3)  # [B, P, Dx]
        out = out + highway
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc)
        dec_out = self.projection(enc_out)
        return dec_out
