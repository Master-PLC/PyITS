import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Decoders import OutputBlock
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class FeatureReuse(nn.Module):
    def __init__(self, d_model, seq_len, r=None):
        super().__init__()

        d_ff = seq_len // 4 if r is None else seq_len // r

        self.conv = nn.Conv1d(d_model, 1, kernel_size=3, padding=1)
        self.W1 = nn.Linear(seq_len, d_ff)
        self.W2 = nn.Linear(d_ff, seq_len)

    def forward(self, x):
        # x shape: [B, L, d_model]
        score = self.conv(x.permute(0, 2, 1)).squeeze(1)  # [B, L]
        score = F.relu(self.W1(score))  # [B, d_ff]
        score = F.sigmoid(self.W2(score))  # [B, L]
        x = torch.einsum('bld, bl->bld', x, score)
        return x


class Model(nn.Module):
    """DLformer: A Dynamic Length Transformer-Based  Network for Efficient Feature Representation in Remaining Useful Life Prediction
    Paper link: https://ieeexplore.ieee.org/abstract/document/10078910/
    """
    supported_tasks = ['rul_estimation']

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.num_seq = configs.num_seq
        self.sub_seq_len = [int(np.ceil(self.seq_len / self.num_seq * i)) for i in range(1, self.num_seq + 1)]
        self.cusum_seq_len = [sum(self.sub_seq_len[:i+1]) for i in range(self.num_seq)]
        self.confidence_threshold = configs.confidence_threshold

        # Embedding
        self.enc_embedding = nn.ModuleList([
            DataEmbedding(
                self.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            ) for _ in range(self.num_seq)
        ])
        # Encoder
        self.encoder = nn.ModuleList([
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False, configs.factor, attention_dropout=configs.dropout, 
                                output_attention=configs.output_attention
                            ), 
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model)
            ) for _ in range(self.num_seq)
        ])
        # Feature reuse
        self.feature_reuse = nn.ModuleList([
            FeatureReuse(configs.d_model, self.cusum_seq_len[i]) for i in range(self.num_seq)
        ])
        # Confidence generator
        self.confidence_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, 1),
                nn.Sigmoid()
            ) for _ in range(self.num_seq)
        ])
        # Decoder
        self.projection = nn.ModuleList([
            OutputBlock(
                configs.d_model, configs.c_out, seq_len=self.cusum_seq_len[i], pred_len=configs.pred_len, 
                task_name=self.task_name, dropout=configs.dropout
            ) for i in range(self.num_seq)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc shape: [B, L, D]
        if self.training:
            features = []
            confidences = []
            outputs = []
            for i in range(self.num_seq):
                x_enc_i = x_enc[:, -self.sub_seq_len[i]:, :]
                x_mark_enc_i = x_mark_enc[:, -self.sub_seq_len[i]:, :] if x_mark_enc is not None else None
                x_enc_i = self.enc_embedding[i](x_enc_i, x_mark_enc_i)

                enc_out_i, attns = self.encoder[i](x_enc_i, attn_mask=None)
                features.append(enc_out_i)
                dec_in_i = torch.cat(features[:i+1], dim=1)
                dec_in_i = self.feature_reuse[i](dec_in_i)

                confidence = self.confidence_generator[i](dec_in_i[:, -1, :])
                confidences.append(confidence)

                dec_out_i = self.projection[i](dec_in_i)
                outputs.append(dec_out_i)

            confidences = torch.concat(confidences, dim=1)
            outputs = torch.concat(outputs, dim=1)
            outputs = torch.concat([outputs, confidences], dim=-1)
            return outputs

        else:
            B, L, D = x_enc.size()

            dec_out = []
            for b in range(B):
                features = []
                confidences = []
                outputs = []
                for i in range(self.num_seq):
                    x_enc_i = x_enc[b:b+1, -self.sub_seq_len[i]:, :]  # [1, Li, D]
                    x_mark_enc_i = x_mark_enc[b:b+1, -self.sub_seq_len[i]:, :] if x_mark_enc is not None else None
                    x_enc_i = self.enc_embedding[i](x_enc_i, x_mark_enc_i)  # [1, Li, d_model]

                    enc_out_i, attns = self.encoder[i](x_enc_i, attn_mask=None)  # [1, Li, d_model]
                    features.append(enc_out_i)
                    dec_in_i = torch.cat(features[:i+1], dim=1)  # [1, L0+L1+...+Li, d_model]
                    dec_in_i = self.feature_reuse[i](dec_in_i)  # [1, L0+L1+...+Li, d_model]

                    confidence = self.confidence_generator[i](dec_in_i[:, -1, :])  # [1, 1]
                    confidences.append(confidence.item())

                    dec_out_i = self.projection[i](dec_in_i)  # [B, 1]
                    outputs.append(dec_out_i)
                    if confidence.item() > self.confidence_threshold:
                        dec_out.append(dec_out_i)
                        break

                if len(dec_out) == b:
                    max_confidence_idx = np.argmax(confidences)
                    dec_out.append(outputs[max_confidence_idx])

            dec_out = torch.cat(dec_out, dim=0)
            return dec_out
