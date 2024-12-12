import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _reverse_repeat_tuple, _single

from layers.Decoders import OutputBlock

EPS = torch.finfo(torch.float32).eps

# @torch.jit.script
def efficient_linterpolate(
    x, offsets, kernel_size, dilation, stride, dilated_positions=None, unconstrained=False
):
    device = x.device

    kernel_rfield = dilation*(kernel_size-1)+1
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield-1, kernel_size, dtype=offsets.dtype)  # kernel_size
    dilated_positions = dilated_positions.to(device)

    max_t0 = (offsets.shape[-2]-1)*stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=device, dtype=offsets.dtype).unsqueeze(-1)  # out_length x 1
    dilated_offsets_repeated = dilated_positions + offsets

    # batch_size x channels x out_length x kernel_size
    T = t0s + dilated_offsets_repeated
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s+torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    with torch.no_grad():
        U = torch.floor(T).to(torch.long)  # 1 x 1 x length x kernel_rfield
        U = torch.clamp(U, min=0, max=x.shape[2]-2)

        U = torch.stack([U, U+1], dim=-1)
        if U.shape[1] < x.shape[1]:
            U = U.repeat(1, x.shape[1], 1, 1, 1)

    x = x.unsqueeze(-1).repeat(1, 1, 1, U.shape[-1])
    x = torch.stack([x.gather(index=U[:, :, :, i, :], dim=-2) for i in range(U.shape[-2])], dim=-1)

    # batch_size x groups x out_length x kernel_rfield x kernel_size
    G = torch.max(torch.zeros(U.shape, device=device), 1-torch.abs(U-T.unsqueeze(-1)))

    mx = torch.multiply(G, x.moveaxis(-2, -1))

    # .float()  # batch_size x channels x output_length x kernel size
    return torch.sum(mx, axis=-1)


class DeformConv1d(nn.Module):
    """Adapted from https://github.com/jwr1995/dc1d/tree/main
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding="valid", dilation=1, groups=1, bias=True, 
        padding_mode="reflect", interpolation_function=efficient_linterpolate, unconstrained= None,  # default None to maintain backwards compatibility
        *args, **kwargs
    ) -> None:
        """1D Deformable convolution kernel layer

        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int) = 1
            bias (bool) = True
            padding_mode: See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
        """
        self.interpolation_function = interpolation_function
        padding_ = padding if isinstance(padding, str) else _single(padding)
        stride_ = _single(stride)
        dilation_ = _single(dilation)
        kernel_size_ = _single(kernel_size)

        super(DeformConv1d, self).__init__(*args, **kwargs)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError("Invalid padding string {!r}, should be one of {}".format(padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride_):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_  # note this is tuple-like for compatibility
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding == 'same':
                for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))

        self.dilated_positions = torch.linspace(0, dilation*kernel_size-dilation, kernel_size)  # automatically store dilation offsets

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        if not unconstrained == None:
            self.unconstrained = unconstrained

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DeformConv1d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def forward(self, input, offsets, mask=None):
        """Forward pass of 1D deformable convolution layer

        Args:
            input (torch.Tensor[batch_size, in_channels, length]): input tensor
            offset (torch.Tensor[batch_size, offset_groups, output length, kernel_size]):
                offsets to be applied for each position in the convolution kernel. Offset groups can be 1 or such that (in_channels%offset_groups == 0) is satisfied.
            mask (torch.Tensor[batch_size, offset_groups, kernel_width, 1, out_width]): To be implemented

        Returns:
            output (torch.Tensor[batch_size, in_channels, length]): output tensor
        """
        in_shape = input.shape
        if self.padding_mode != 'zeros':
            input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode
            )

        elif self.padding == 'same':
            input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode='constant',
                value=0
            )

        if "unconstrained" in self.__dict__.keys():
            input = self.interpolation_function(
                input,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                unconstrained=self.unconstrained
            )
        else:
            input = self.interpolation_function(
                input,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions
            )

        input = input.flatten(-2, -1)
        output = F.conv1d(input, self.weight, self.bias, stride=self.kernel_size, groups=self.groups)
        if self.padding == 'same':
            assert in_shape[-1] == output.shape[-1], f"input length {in_shape} and output length {output.shape} do not match."
        return output


class PackedDeformConv1d(DeformConv1d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding="valid", dilation=1, groups=1, bias=True,
        padding_mode="reflect", offset_groups=1, interpolation_function=efficient_linterpolate, unconstrained=None, # default None to maintain backwards compatibility
        *args, **kwargs
    ) -> None:
        """Packed 1D Deformable convolution class. Depthwise-Separable convolution is used to compute offsets.

        Args:
            in_channels (int): Value of convolution kernel size
            out_channels (int): Value of convolution kernel dilation factor
            kernel_size (int): Value of convolution kernel size
            stride (int): Value convolution kernel stride
            padding (int): See torch.nn.Conv1d for details. Default "valid". Still experimental beware of unexpected behaviour.
            dilation (int): Value of convolution kernel dilation factor
            groups (int): 1 or in_channels
            bias (bool): Whether to use bias. Default = True
            padding_mode (str): See torch.nn.Conv1d for details. Default "reflect". Still experimental beware of unexpected behaviour.
            offset_groups (int): 1 or in_channels
        """
        assert offset_groups in [1, in_channels], "offset_groups only implemented for offset_groups in {1,in_channels}"

        super(PackedDeformConv1d,self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, interpolation_function=interpolation_function, 
            unconstrained=unconstrained, *args, **kwargs
        )
        self.offset_groups = offset_groups

        self.offset_dconv = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride=1, groups=in_channels, padding=padding, padding_mode=padding_mode, bias=False
        )
        self.odc_norm = gLN(in_channels)
        self.odc_prelu = nn.PReLU()

        self.offset_pconv = nn.Conv1d(in_channels, kernel_size*offset_groups, 1, stride=1, bias=False)
        self.odp_norm = gLN(kernel_size*offset_groups)
        self.odp_prelu = nn.PReLU()

    def forward(self, input, with_offsets=False):
        """Forward pass of 1D deformable convolution layer

        Args:
            input (Tensor[batch_size, in_channels, length]): input tensor

        Returns:
            output (Tensor[batch_size, in_channels, length]): output tensor
        """
        offsets = self.offset_dconv(input)
        offsets = self.odc_norm(self.odc_prelu(offsets).moveaxis(1,2)).moveaxis(2,1)

        offsets = self.offset_pconv(offsets)
        offsets = self.odp_norm(self.odp_prelu(offsets).moveaxis(1,2)).moveaxis(2,1)  # batch_size x (kernel_size*offset_groups) x length
        offsets = offsets.unsqueeze(0).chunk(self.offset_groups,dim=2)  # batch_size x offset_groups x length x kernel_size
        offsets = torch.vstack(offsets).moveaxis((0,2),(1,3))  # batch_size x offset_groups x length x kernel_size

        if with_offsets:
            return super().forward(input, offsets), offsets
        else:
            return super().forward(input, offsets)


class gLN(nn.Module):
    """Global Layer Normalization (gLN).

    Copyright SpeechBrain 2022

    Arguments
    ---------
    channel_size 
        Number of channels in the third dimension.

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = GlobalLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(gLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Arguments
        ---------
        y : Tensor
            Tensor shape [M, K, N]. M is batch size, N is channel size, and K is length.

        Returns
        -------
        gLN_y : Tensor
            Tensor shape [M, K. N]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class cLN(nn.Module):
    """Channel-wise Layer Normalization (cLN).

    Arguments
    ---------
    channel_size 
        Number of channels in the normalization dimension (the third dimension).

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = ChannelwiseLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(cLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, K, N]
        """
        mean = torch.mean(y, dim=2, keepdim=True)  # [M, K, 1]
        var = torch.var(y, dim=2, keepdim=True, unbiased=False)  # [M, K, 1]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class ChannelSelection(nn.Module):
    def __init__(self, d_model, r=None, **kwargs):
        super(ChannelSelection, self).__init__()

        d_ff = d_model // r if r is not None else d_model // 4
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fce = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
        )
        self.fcw = nn.Sequential(
            nn.Linear(d_ff, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, D, L]
        score = self.global_pool(x).squeeze(-1)  # [B, D]
        score = self.fce(score)  # [B, d_ff]
        score = self.fcw(score)  # [B, D]
        x = torch.einsum("bdl,bd->bdl", x, score)
        return x


class Model(nn.Module):
    """AdaNet: An Adaptive and Dynamical Neural Network for  Machine Remaining Useful Life Prediction
    Paper link: https://ieeexplore.ieee.org/document/10065450
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_layers = configs.e_layers
        self.output_attention = configs.output_attention

        self.DCs = nn.ModuleList([
            PackedDeformConv1d(
                in_channels=configs.enc_in if i == 0 else configs.d_model, out_channels=configs.d_model, 
                kernel_size=configs.kernel_size, stride=1, padding="same", bias=True
            ) for i in range(configs.e_layers)
        ])
        self.BNs = nn.ModuleList([
            nn.BatchNorm1d(configs.d_model) for i in range(configs.e_layers)
        ])
        self.CSs = nn.ModuleList([
            ChannelSelection(configs.d_model) for i in range(configs.e_layers)
        ])

        self.projection = OutputBlock(
            configs.d_model, configs.c_out, seq_len=configs.seq_len, pred_len=self.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        # x_enc: [B, L, D]
        dec_in = x_enc.transpose(1, 2)  # [B, D, L]
        for i in range(self.num_layers):
            dec_in = self.DCs[i](dec_in)  # [B, d_model, L]
            dec_in = self.BNs[i](dec_in)  # [B, d_model, L]
            dec_in = self.CSs[i](dec_in)  # [B, d_model, L]
            dec_in = F.relu(dec_in)
        dec_in = dec_in.transpose(1, 2)  # [B, L, d_model]
        dec_out = self.projection(dec_in)
        return dec_out
