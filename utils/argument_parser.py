import argparse
import os
import sys

from utils.tools import EvalAction, load_yaml


def parse_cmd_arguments():
    parser = argparse.ArgumentParser(description='UniMonitor')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='soft_sensor',
                        help='task name, options: [soft_sensor, fault_diagnosis, process_monitoring, process_maintenance, rul_estimation]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [Autoformer, iTransformer, TimesNet]')
    parser.add_argument('--fix_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--rerun', action='store_true', help='rerun', default=False)

    # save
    parser.add_argument('--save_root', type=str, default='./output/', help='root path of the results')
    parser.add_argument('--remove_log', action='store_false', help='remove log', default=True)
    parser.add_argument('--output_pred', action='store_true', help='output true and pred', default=False)
    parser.add_argument('--output_vis', action='store_true', help='output visual figures', default=False)

    # data loader
    parser.add_argument('--data', type=str, required=True, default='SRU', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/SRU', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='SRU_data.txt', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target_idx', default=[0], action=EvalAction, help='target index')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--shift', type=int, default=0, help='shift of target as features')
    parser.add_argument('--data_percentage', type=float, default=1., help='percentage of training data')
    parser.add_argument('--scale', default=True, action='store_false', help='scale data')

    # task define
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length for model')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=7, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    parser.add_argument('--conv_kernel', default=[12,16], action=EvalAction, help='downsampling and upsampling convolution kernel_size')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=12,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--individual', default=False, action=EvalAction, help='whether shared model among different variates.')
    parser.add_argument('--version', type=str, default='Fourier', help='version of FEDformer')
    parser.add_argument('--mode_select', type=str, default='random', help='for FEDformer, there are two mode selection method, options: [random, low].')
    parser.add_argument('--modes', type=int, default=32, help='modes to be selected for FEDformer.')
    parser.add_argument('--num_blocks', type=int, default=3, help='number of Koopa blocks.')
    parser.add_argument('--multistep', default=False, action=EvalAction, help='whether to use approximation for multistep K.')
    parser.add_argument('--chunk_size', type=int, default=24, help='reshape T into [num_chunks, chunk_size]')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length for PAttn')
    parser.add_argument('--stride', type=int, default=8, help='stride for PAttn')
    parser.add_argument('--window_size', default=[4, 4], action=EvalAction, help='the downsample window size in pyramidal attention.')
    parser.add_argument('--inner_size', type=int, default=5, help='the size of neighbour attention.')
    parser.add_argument('--bucket_size', type=int, default=4, help='bucket size for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='hashes for Reformer')
    parser.add_argument('--feature_encode_dim', type=int, default=2, help='feature encoding dimension for TiDE')
    parser.add_argument('--bias', default=True, action='store_false', help='bias')
    parser.add_argument('--cut_freq', type=int, default=5)
    parser.add_argument('--num_experts_list', default=[4, 4, 4], action=EvalAction)
    parser.add_argument('--patch_size_list', default=[16,12,8,32,12,8,6,4,8,6,4,2], action=EvalAction)
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--mem_dim', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size for convolution')
    parser.add_argument('--coef', type=float, default=1.0, help='any coefficient')
    parser.add_argument('--num_seq', type=int, default=3, help='number of sub sequences')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='confidence threshold for DLformer')
    parser.add_argument('--d_lower', type=int, default=128, help='lower dimension for LinearAttention')
    parser.add_argument('--kernel_type', type=str, default='WFK', help='kernel type for MCN')
    parser.add_argument('--n_kernels', type=int, default=8, help='number of kernels for MCN')

    # ml model define
    parser.add_argument('--lv_dimensions', default=[100, 150], action=EvalAction, help='Dimension of latent variables in each PLS layer.')
    parser.add_argument('--pls_solver', type=str, default='svd', help='Solver type of the PLS algorithm.')
    parser.add_argument('--use_nonlinear_mapping', default=True, action=EvalAction, help='Whether to use nonlinear mapping or not.')
    parser.add_argument('--mapping_dimensions', default=[128, 128], action=EvalAction, help='Dimension of nonlinear features in each nonlinear mapping layer.')
    parser.add_argument('--nys_gamma_values', default=[0.014, 2.8], action=EvalAction, help='Gamma values of Nystroem function in each nonlinear mapping layer. Only effective when use_nonlinear_mapping == True.')
    parser.add_argument('--stack_previous_lv1', default=True, action=EvalAction, help='Whether to stack the first latent variable of the previous PLS layer into the current nonlinear features.')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--delta', type=float, default=0., help='delta for early stopping')
    parser.add_argument('--metric_mode', type=str, default='min', help='metric mode')
    parser.add_argument('--lr_mode', type=str, default='min', help='lr mode')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='min lr')
    parser.add_argument('--step_size', type=int, default=2, help='lr decay step')

    # regularization
    parser.add_argument('--rec_lambda', type=float, default=1., help='weight of reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=0., help='weight of auxilary function')

    # GPU
    parser.add_argument('--gpu_ids', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # for solver
    parser.add_argument('--solver', type=str, default='linear')
    parser.add_argument('--loss_scale', default=None, action=EvalAction)
    parser.add_argument("--iteration_window", default=25, type=int)
    parser.add_argument("--temp", default=2., type=float)
    parser.add_argument("--gamma", default=0.01, type=float)
    parser.add_argument("--w_lr", default=0.025, type=float)
    parser.add_argument("--max_norm", default=4., type=float)
    parser.add_argument("--alpha", default=1.5, type=float)
    parser.add_argument("--params", type=str, default="shared")
    parser.add_argument("--normalization", type=str, default="loss+")
    parser.add_argument("--optim_niter", default=20, type=int)
    parser.add_argument("--update_weights_every", default=1, type=int)
    parser.add_argument("--cmin", type=float, default=0.2)
    parser.add_argument("--c", default=0.4, type=float)
    parser.add_argument("--rescale", default=1, type=int)
    parser.add_argument("--rank", default=10, type=int)
    parser.add_argument("--num_chunk", default=10, type=int)
    parser.add_argument("--n_sample_group", default=4, type=int)
    parser.add_argument("--grad_reduction", default="mean", type=str)

    # for MoEs
    parser.add_argument("--n_exp", default=3, type=int)
    parser.add_argument("--n_exp_shared", default=3, type=int)
    parser.add_argument("--exp_layer", default=2, type=int)
    parser.add_argument("--tower_layer", default=1, type=int)
    parser.add_argument("--exp_hidden", default=128, type=int)
    parser.add_argument("--exp_type", default="mlp", type=str)
    parser.add_argument("--gate_type", default="softmax", type=str)
    parser.add_argument("--output_type", default="moe", type=str)
    parser.add_argument("--init_ratio", default=0.1, type=float)

    # for predictive maintenance
    parser.add_argument('--anomaly_threshold', type=float, default=0.1, help='anomaly threshold')

    args, _ = parser.parse_known_args()

    return vars(args)


def parse_arguments():
    # priority: input "args" dict > YAML file
    cmd_args = parse_cmd_arguments()

    # default args in UniMonitor
    config_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
    config_dir = f"{cmd_args['data']}/{cmd_args['task_name']}/{cmd_args['model']}"
    if os.path.exists(os.path.join(config_base_dir, config_dir)):
        print(f"Load configuration files from {config_dir}")
        config = load_yaml(os.path.join(config_base_dir, config_dir, 'config.yaml'))
    else:
        config = {}

    config.update(cmd_args)

    sys_args = sys.argv[1:]
    grouped_args, temp_group = [], []
    for i in range(len(sys_args)):
        if sys_args[i].startswith('--'):
            if temp_group:
                grouped_args.append(temp_group)
            temp_group = [sys_args[i][2:]]
        else:
            temp_group.append(sys_args[i])
    if temp_group:
        grouped_args.append(temp_group)

    filtered_args = [
        'is_training', 'rerun', 'remove_log', 'output_pred', 'output_vis', 'inverse',
        'save_root', 'root_path', 'model', 'data', 'task_name', 'data_path'
    ]

    # Too long to makedir
    # grouped_args = [arg_group for arg_group in grouped_args if arg_group[0] not in filtered_args]
    # sys_args = sys.argv[:1] + [arg for arg_group in grouped_args for arg in arg_group]

    grouped_args = [arg_group[1] for arg_group in grouped_args if arg_group[0] not in filtered_args]
    sys_args = [sys.argv[0].split('/')[-1]] + grouped_args
    setting = '_'.join(sys_args)

    config['setting'] = setting
    config['save_dir'] = os.path.join(config['save_root'], config['data'], config['task_name'], config['model'], setting)
    os.makedirs(config['save_dir'], exist_ok=True)

    return argparse.Namespace(**config)
