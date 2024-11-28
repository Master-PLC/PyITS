import math

import torch
import torch.nn as nn

from data_provider import DATA_DICT
from layers.Decoders import OutputBlock


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """

    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum  # [alpha * (L//2+1)]

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)  # [B, L//2+1, D]
        mask = torch.ones_like(xf)  # [B, L//2+1, D]
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf*mask, dim=1)  # [B, L, D]
        x_inv = x - x_var

        return x_var, x_inv


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self, f_in, f_out, hidden_dim=128, hidden_layers=2, dropout=0.05, activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), self.activation, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: B x S x f_in, y: B x S x f_out
        y = self.layers(x)
        return y


class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """

    def __init__(self):
        super(KPLayer, self).__init__()

        self.K = None  # [B, d_model, d_model]

    def one_step_forward(self, z, return_rec=False, return_K=False):
        # z: [B, F, d_model]
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]  # [B, F-1, d_model], [B, F-1, d_model]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # [B, d_model, d_model]
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)  # [B, 1, d_model] * [B, d_model, d_model] -> [B, 1, d_model]
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # [B, 1, d_model] + [B, F-1, d_model] * [B, d_model, d_model] -> [B, F, d_model]
            return z_rec, z_pred  # [B, F, d_model], [B, 1, d_model]

        return z_pred

    def forward(self, z, pred_len=1):
        # z: [B, F, d_model]
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)  # [B, F, d_model], [B, 1, d_model]
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)  # [B, 1, d_model] * [B, d_model, d_model] -> [B, 1, d_model]
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)  # [B, step, d_model]
        return z_rec, z_preds  # [B, F, d_model], [B, step, d_model]


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """

    def __init__(self):
        super(KPLayerApprox, self).__init__()

        self.K = None  # [B, d_model, d_model]
        self.K_step = None  # [B, d_model, d_model]

    def forward(self, z, pred_len=1):
        # z: [B, F, d_model]
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]  # [B, F-1, d_model], [B, F-1, d_model]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # [B, d_model, d_model]

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # [B, 1, d_model] + [B, F-1, d_model] * [B, d_model, d_model] -> [B, F, d_model]

        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)  # [B, d_model, d_model]
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)  # [B, step, d_model] * [B, d_model, d_model] -> [B, step, d_model]

        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred  # [B, F, d_model], [B, step, d_model]


class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """

    def __init__(
        self, enc_in=8, input_len=96, pred_len=96, seg_len=24, dynamic_dim=128, encoder=None, decoder=None, multistep=False
    ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep

        self.encoder, self.decoder = encoder, decoder
        # segment number of input
        self.freq = math.ceil(self.input_len / self.seg_len)
        # segment number of output
        self.step = math.ceil(self.pred_len / self.seg_len)
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer()

    def forward(self, x):
        # x: [B, L, D]
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x), dim=1)

        res = res.chunk(self.freq, dim=1)  # F * [B, seg_len, D]
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # [B, F, seg_len*D]

        res = self.encoder(res)  # [B, F, d_model]
        x_rec, x_pred = self.dynamics(res, self.step)  # [B, F, d_model], [B, step, d_model]

        x_rec = self.decoder(x_rec)  # [B, F, seg_len*D]
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)  # [B, F, seg_len, D]
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # [B, L, D]

        x_pred = self.decoder(x_pred)  # [B, step, seg_len*D]
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)  # [B, step, seg_len, D]
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # [B, P, D]

        return x_rec, x_pred  # [B, L, D], [B, P, D]


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """

    def __init__(self, input_len=96, pred_len=96, dynamic_dim=128, encoder=None, decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init)  # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        # x: [B, L, D]
        res = x.transpose(1, 2)  # [B, D, L]
        res = self.encoder(res)  # [B, D, d_model]
        res = self.K(res)  # [B, D, d_model]
        res = self.decoder(res)  # [B, D, P]
        res = res.transpose(1, 2)  # [B, P, D]

        return res


class Model(nn.Module):
    '''
    Paper link: https://arxiv.org/pdf/2305.18803.pdf
    '''
    supported_tasks = ['soft_sensor', 'process_monitoring', 'fault_diagnosis', 'rul_estimation', 'predictive_maintenance']

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.seq_len // 2

        self.seg_len = self.pred_len
        self.num_blocks = configs.num_blocks
        self.dynamic_dim = configs.d_model
        self.hidden_dim = configs.d_ff
        self.hidden_layers = configs.e_layers
        self.multistep = configs.multistep
        self.alpha = 0.2

        self.mask_spectrum = self._get_mask_spectrum(configs)
        self.disentanglement = FourierFilter(self.mask_spectrum)

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(
            f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers
        )
        self.time_inv_decoder = MLP(
            f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers
        )
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([
            TimeInvKP(
                input_len=self.input_len, pred_len=self.pred_len, dynamic_dim=self.dynamic_dim,
                encoder=self.time_inv_encoder, decoder=self.time_inv_decoder
            ) for _ in range(self.num_blocks)
        ])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(
            f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='tanh',
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers
        )
        self.time_var_decoder = MLP(
            f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='tanh',
            hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers
        )
        self.time_var_kps = nn.ModuleList([
            TimeVarKP(
                enc_in=configs.enc_in, input_len=self.input_len, pred_len=self.pred_len, 
                seg_len=self.seg_len, dynamic_dim=self.dynamic_dim, encoder=self.time_var_encoder,
                decoder=self.time_var_decoder, multistep=self.multistep
            ) for _ in range(self.num_blocks)
        ])

        # Decoder
        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=self.pred_len, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def _get_mask_spectrum(self, configs):
        """get shared frequency spectrums
        """
        dataset = DATA_DICT[configs.data](configs, None)
        dataset.generate_data(task_name=configs.task_name)
        train_data = dataset.get_data(flag='train')

        lookback_window = torch.tensor(train_data[0])  # [N, L, D]
        amps = abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)  # [N, L//2+1, D] -> [L//2+1]
        mask_spectrum = amps.topk(int(amps.shape[0] * self.alpha)).indices  # [alpha * (L//2+1)]
        return mask_spectrum  # as the spectrums of time-invariant component

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # [B, 1, Dx]
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # [B, 1, Dx]
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)  # [B, L, Dx], [B, L, Dx]
            time_inv_output = self.time_inv_kps[i](time_inv_input)  # [B, P, Dx]
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)  # [B, L, Dx], [B, P, Dx]
            residual = residual - time_var_backcast  # [B, L, Dx]
            if forecast is None:
                forecast = (time_inv_output + time_var_output)  # [B, P, Dx]
            else:
                forecast += (time_inv_output + time_var_output)

        # Series Stationarization adopted from NSformer
        res = forecast * std_enc + mean_enc  # [B, P, Dx]
        res = self.projection(res)
        return res
