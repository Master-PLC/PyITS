import torch
import torch.nn as nn

from layers.Decoders import OutputBlock
from layers.Pathformer_EncDec import CustomLinear, WeightGenerator


def prime_factors(n):
    factors = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    factors.sort(reverse=True)
    return factors


class Model(nn.Module):
    """Paper link: https://arxiv.org/abs/2204.13767
    """
    supported_tasks = ['soft_sensor', 'process_monitoring', 'rul_estimation', 'fault_diagnosis', 'predictive_maintenance']

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.factorized = True
        self.num_nodes = configs.enc_in
        self.output_dim = configs.c_out
        self.channels = configs.d_model
        self.horizon = configs.seq_len
        self.lag = configs.seq_len
        self.patch_sizes = prime_factors(self.lag)

        self.start_fc = nn.Linear(in_features=1, out_features=self.channels)
        cuts = self.lag
        self.layers = nn.ModuleList()
        self.skip_generators = nn.ModuleList()

        for patch_size in self.patch_sizes:
            if cuts % patch_size != 0:
                raise Exception('Lag not divisible by patch size')

            cuts = int(cuts / patch_size)
            self.layers.append(
                Layer(
                    input_dim=self.channels, num_nodes=self.num_nodes, cuts=cuts,
                    cut_size=patch_size, factorized=self.factorized
                )
            )
            self.skip_generators.append(
                WeightGenerator(
                    in_dim=cuts * self.channels, out_dim=256, number_of_weights=1,
                    mem_dim=configs.mem_dim, num_nodes=self.num_nodes, factorized=False
                )
            )

        self.custom_linear = CustomLinear(factorized=False)

        self.projection = OutputBlock(
            configs.enc_in, configs.c_out, seq_len=256, pred_len=configs.pred_len, 
            task_name=self.task_name, dropout=configs.dropout
        )

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        x = self.start_fc(batch_x.unsqueeze(-1))  # [B, L, Dx, d_model]
        batch_size = x.size(0)

        skip = 0
        for layer, skip_generator in zip(self.layers, self.skip_generators):
            x = layer(x)
            weights, biases = skip_generator()
            skip_inp = x.transpose(2, 1).reshape(batch_size, 1, self.num_nodes, -1)
            skip = skip + self.custom_linear(skip_inp, weights[-1], biases[-1])

        x = torch.relu(skip).squeeze(1)
        x = self.projection(x.transpose(1, 2))
        return x


class Layer(nn.Module):
    def __init__(self, input_dim, num_nodes, cuts, cut_size, factorized):
        super(Layer, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes

        self.cuts = cuts
        self.cut_size = cut_size
        self.temporal_embeddings = nn.Parameter(torch.rand(cuts, 1, 1, self.num_nodes, 5))

        self.embeddings_generator = nn.ModuleList([
            nn.Sequential(*[nn.Linear(5, input_dim)]) for _ in range(cuts)
        ])

        self.out_net1 = nn.Sequential(*[
            nn.Linear(input_dim, input_dim ** 2),
            nn.Tanh(),
            nn.Linear(input_dim ** 2, input_dim),
            nn.Tanh(),
        ])

        self.out_net2 = nn.Sequential(*[
            nn.Linear(input_dim, input_dim ** 2),
            nn.Tanh(),
            nn.Linear(input_dim ** 2, input_dim),
            nn.Sigmoid(),
        ])

        self.temporal_att = TemporalAttention(input_dim, factorized=factorized)
        self.weights_generator_shared = WeightGenerator(
            input_dim, input_dim, mem_dim=None, num_nodes=num_nodes, factorized=False, number_of_weights=2
        )
        self.weights_generator_distinct = WeightGenerator(
            input_dim, input_dim, mem_dim=5, num_nodes=num_nodes, factorized=factorized, number_of_weights=2
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, L, Dx, d_model]
        batch_size = x.size(0)

        data_concat = None
        out = 0

        weights_shared, biases_shared = self.weights_generator_shared()  # 2 * [d_model, d_model], 2 * [1, d_model]
        weights_distinct, biases_distinct = self.weights_generator_distinct()  # 2 * [Dx, d_model, d_model], 2 * [Dx, d_model]

        for i in range(self.cuts):
            # shape is (B, cut_size, N, C)
            t = x[:, i * self.cut_size:(i + 1) * self.cut_size, :, :]  # [B, cut_size, Dx, d_model]

            if i != 0:
                out = self.out_net1(out) * self.out_net2(out)

            emb = self.embeddings_generator[i](self.temporal_embeddings[i]).repeat(batch_size, 1, 1, 1) + out
            t = torch.cat([emb, t], dim=1)
            out = self.temporal_att(t[:, :1, :, :], t, t, weights_distinct, biases_distinct, weights_shared, biases_shared)

            if data_concat == None:
                data_concat = out
            else:
                data_concat = torch.cat([data_concat, out], dim=1)

        return self.dropout(data_concat)


class TemporalAttention(nn.Module):
    def __init__(self, in_dim, factorized):
        super(TemporalAttention, self).__init__()
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]

        # [batch_size, num_step, N, K * head_size]
        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        # query: [K * batch_size, N, 1, head_size]
        # key:   [K * batch_size, N, head_size, num_step]
        # value: [K * batch_size, N, num_step, head_size]
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        # normalize the attention scores
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.tanh(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x
