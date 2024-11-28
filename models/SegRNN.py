import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Decoders import OutputBlock


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.task_name = configs.task_name
        self.pred_len = configs.seq_len

        self.seg_len = configs.seg_len
        if self.seq_len % self.seg_len == 0:
            self.seg_num_x = self.seq_len // self.seg_len
            self.input_padding = nn.Identity()
        else:
            self.seg_num_x = self.seq_len // self.seg_len + 1
            self.input_padding = nn.ConstantPad1d((self.seg_num_x * self.seg_len - self.seq_len, 0), 0)

        if self.pred_len % self.seg_len == 0:
            self.seg_num_y = self.pred_len // self.seg_len
        else:
            self.seg_num_y = self.pred_len // self.seg_len + 1

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(
            input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
            batch_first=True, bidirectional=False
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def encoder(self, x):
        # x: [B, L, Dx]
        batch_size = x.size(0)

        # normalization and permute
        seq_last = x[:, -1:, :].detach()  # [B, 1, Dx]
        x = (x - seq_last).permute(0, 2, 1)  # [B, Dx, L]

        # segment and embedding
        x = self.input_padding(x)  # [B, Dx, Ni*S]
        x = x.reshape(-1, self.seg_num_x, self.seg_len)  # [B*Dx, Ni, S]
        x = self.valueEmbedding(x)  # [B*Dx, Ni, d_model]

        # encoding
        _, hn = self.rnn(x)  # [1, B*Dx, d_model]

        pos_emb = torch.cat(
            [
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # [Dx, No, d_model//2]
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # [Dx, No, d_model//2]
            ], dim=-1
        ).view(-1, 1, self.d_model)  # [Dx*No, 1, d_model]
        pos_emb = pos_emb.repeat(batch_size,1,1)  # [B*Dx*No, 1, d_model]

        _, hy = self.rnn(
            pos_emb,  # [B*Dx*No, 1, d_model]
            hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)  # [1, B*Dx*No, d_model]
        )  # [1, B*Dx*No, d_model]

        y = self.predict(hy)  # [1, B*Dx*No, S]
        y = y.view(-1, self.enc_in, self.seg_num_y * self.seg_len)  # [B, Dx, No*S]

        # permute and denorm
        y = y.permute(0, 2, 1)[:, -self.pred_len:, :]  # [B, P, Dx]
        y = y + seq_last
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Encoder
        enc_out = self.encoder(x_enc)
        dec_out = self.projection(enc_out)
        return dec_out
