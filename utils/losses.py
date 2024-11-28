import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          torch.abs(forecast.data) + torch.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


class CumulLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CumulLoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(self, outputs, labels):
        # Compute the cumulative sum along the sequence dimension (dim=1)
        cum_outputs = torch.cumsum(outputs, dim=1)
        cum_labels = torch.cumsum(labels, dim=1)

        # Compute the MAE loss between the cumulative sums of outputs and labels
        loss = self.mae_loss(cum_outputs, cum_labels)
        return loss


class WMSELoss(nn.Module):
    def __init__(self):
        super(WMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, labels):
        loss = self.mse_loss(outputs, labels)

        error = outputs - labels
        coef = torch.where(
            error >= 0,
            1.5 - 2 * torch.exp(-error) / (1 + torch.exp(-error))**2,
            1.5 - 2 * torch.exp(-0.5 * error) / (1 + torch.exp(-0.5 * error))**2
        )

        loss = (loss * coef).mean()
        return loss


class ConfidenceLoss(nn.Module):
    def __init__(self, num_seq, alpha):
        super(ConfidenceLoss, self).__init__()
        self.num_seq = num_seq
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, labels):
        outputs, confidences = outputs.chunk(2, dim=-1)
        loss = 0
        for i in range(self.num_seq):
            loss += self.mse_loss(outputs[:, i:i+1], labels)
            confidence_label = F.tanh((outputs[:, i:i+1] - labels).abs() / self.alpha)
            loss += self.mse_loss(confidences[:, i:i+1], confidence_label)
        return loss
