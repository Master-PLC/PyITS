import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(
        self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24], isometric_kernel=[18, 6]
    ):
        super().__init__()
        self.conv_kernel = conv_kernel

        # isometric convolution
        self.isometric_conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=1
            ) for i in isometric_kernel
        ])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=i // 2, stride=i
            ) for i in conv_kernel
        ])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([
            nn.ConvTranspose1d(
                in_channels=feature_size, out_channels=feature_size, kernel_size=i, padding=0, stride=i
            ) for i in conv_kernel
        ])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(
            in_channels=feature_size, out_channels=feature_size, kernel_size=(len(self.conv_kernel), 1)
        )

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)  # [B, d_model, L]

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))  # [B, d_model, L']
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=x.device)  # [B, d_model, L'-1]
        x = torch.cat((zeros, x), dim=-1)  # [B, d_model, 2L'-1]
        x = self.drop(self.act(isometric(x)))  # [B, d_model, L'']
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)  # [B, d_model, L']

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))  # [B, d_model, L*]
        x = x[:, :, :seq_len]  # truncate [B, d_model, L]

        x = self.norm(x.permute(0, 2, 1) + input)  # [B, L, d_model]
        return x

    def forward(self, src):
        # src: [B, L, d_model]
        multi = []  # multi-scale
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)  # [B, L, d_model], [B, L, d_model]
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])  # [B, L, d_model]
            multi.append(src_out)

        # merge
        mg = torch.stack(multi, dim=1)  # [B, len(conv_kernel), L, d_model]
        mg = self.merge(mg.permute(0, 3, 1, 2))  # [B, d_model, len(conv_kernel), L] -> [B, d_model, 1, L]
        mg = mg.squeeze(-2).permute(0, 2, 1)  # [B, L, d_model]

        y = self.norm1(mg)  # [B, L, d_model]
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)  # [B, L, d_model]

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(
        self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1, 
        conv_kernel=[2, 4], isometric_kernel=[18, 6]
    ):
        super().__init__()

        self.mic = nn.ModuleList([
            MIC(
                feature_size=embedding_size, n_heads=n_heads, decomp_kernel=decomp_kernel, 
                conv_kernel=conv_kernel, isometric_kernel=isometric_kernel
            ) for i in range(d_layers)
        ])
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=zt53IDUR1U
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()

        conv_kernel = configs.conv_kernel
        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs.seq_len + configs.seq_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((configs.seq_len + configs.seq_len + ii - 1) // ii)

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.conv_trans = SeasonalPrediction(
            embedding_size=configs.d_model, n_heads=configs.n_heads, dropout=configs.dropout,
            d_layers=configs.d_layers, decomp_kernel=decomp_kernel, c_out=configs.enc_in, 
            conv_kernel=conv_kernel, isometric_kernel=isometric_kernel
        )

        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=configs.seq_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)  # [B, L, Dx], [B, L, Dx]

        # embedding
        dec_out = self.dec_embedding(seasonal_init_enc, x_mark_dec)  # [B, L, d_model]
        dec_out = self.conv_trans(dec_out)  # [B, L, Dx]
        dec_out = dec_out + trend
        dec_out = self.projection(dec_out)
        return dec_out
