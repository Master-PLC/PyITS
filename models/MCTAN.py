import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, LocalAttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class ChannelAttention(nn.Module):
    def __init__(self, enc_in, seq_len):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        
        self.Wp = nn.Linear(seq_len, 1)
        self.Wu = nn.Linear(enc_in, 2 * enc_in)
        self.Wd = nn.Linear(2 * enc_in, enc_in)

    def forward(self, x_enc):
        # x_enc shape: [B, L, D]
        output = self.Wp(x_enc.permute(0, 2, 1)).squeeze(-1)  # [B, D]
        output = self.Wu(output)  # [B, 2 * D]
        output = F.relu(output)
        output = self.Wd(output)  # [B, D]
        Ca = F.softmax(output, dim=-1)
        x_enc = torch.einsum('bld, bd->bld', x_enc, Ca)
        return x_enc


class Model(nn.Module):
    """MCTAN: A Novel Multichannel Temporal Attention-Based Network for Industrial Health Indicator Prediction
    Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9675827
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len + 1
        self.output_attention = configs.output_attention

        self.index_token = torch.ones(self.enc_in) * configs.coef  # raw method
        # self.index_token = nn.Parameter(torch.ones(self.enc_in) * configs.coef)  # learnable method

        self.channel_att = ChannelAttention(self.enc_in, self.seq_len)
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LocalAttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ),
                        self.seq_len,
                        configs.d_model,
                        configs.kernel_size,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_model * 2,
                    dropout=configs.dropout,
                    activation="relu"
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU()
        )
        self.projection = OutputBlock(
            configs.d_model // 2, configs.c_out, seq_len=self.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc shape: [B, L, D]
        B, L, D = x_enc.size()
        x_enc = torch.cat([x_enc, self.index_token.repeat(B, 1, 1).to(x_enc.device)], dim=1)  # [B, L + 1, D]
        enc_in = self.channel_att(x_enc)  # [B, L + 1, D]
        enc_in = self.enc_embedding(enc_in, x_mark_enc)  # [B, L + 1, d_model]
        enc_out, attn = self.encoder(enc_in)  # [B, L + 1, d_model]
        dec_out = self.head(enc_out)  # [B, L + 1, d_model // 2]
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out, attn
        return dec_out
