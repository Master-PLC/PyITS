from typing import Union

import torch
from einops import rearrange, repeat
from torch import nn
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


class Model(nn.Module):
    def __init__(
        self,
        config,
        variate_mode: str = 'M',
        patch_size: Union[str, int] = 64,
        model_size: str = 'large',
        scaling: bool = True,
        **kwargs
    ):
        super().__init__()
        self.context_length = config.seq_len
        self.prediction_length = config.pred_len
        self.target_dim = config.target_dim
        self.freq = config.freq
        self.dataset = config.dataset
        self.num_samples = config.num_samples
        self.patch_size = int(config.patch_size)
        
        # Load pretrained model
        self.moirai = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            target_dim=self.target_dim,
            scaling=scaling
        )
        if not config.use_p:
            for param in self.moirai.parameters():
                param.data.uniform_(-0.02, 0.02)

    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec):
        # B, _, K = inputs.shape
        # inputs = rearrange(inputs, 'b l k -> (b k) l 1') #(210,96,1)
        past_observed_values = torch.ones_like(inputs, dtype=torch.bool).to(inputs.device)
        past_is_pad = torch.zeros(inputs.shape[:2], dtype=torch.float32).to(inputs.device)
        forecasts = self.moirai(
            past_target=inputs,
            past_observed_target=past_observed_values,
            past_is_pad=past_is_pad,
            num_samples=self.num_samples
        )
        forecast = forecasts.mean(axis=1)
        # forecast = rearrange(forecast, '(b k) l -> b l k', b=B, k=K)
        return forecast
