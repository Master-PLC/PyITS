import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
from torch import optim
from utils.logger import wrap_message

plt.switch_backend('agg')


class Scheduler:
    def __init__(self, args, optimizer, logger):
        self.logger = logger

        self.optimizer = optimizer
        self.scheduler_type = args.lradj
        self.learning_rate = args.learning_rate
        self.last_lr = optimizer.param_groups[0]['lr']

        if self.scheduler_type is None or self.scheduler_type == 'none':
            self.scheduler = None
        elif self.scheduler_type == 'reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=args.lr_mode, factor=args.lr_decay, patience=args.step_size, min_lr=args.min_lr)
        elif self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.step_size, eta_min=args.min_lr)
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
        else:
            self.scheduler = None

    def step(self, epoch, eval_loss=None):
        if self.scheduler_type == 'type1':
            lr_adjust = {epoch: self.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif self.scheduler_type == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }

        if self.scheduler_type in ['type1', 'type2']:
            if epoch in lr_adjust.keys():
                lr = lr_adjust[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.logger.info(wrap_message(f'Updating learning rate ({self.last_lr} --> {lr}).', color='red'))
                self.last_lr = lr
                return

        if self.scheduler_type is None or self.scheduler_type == 'none':
            return
        elif self.scheduler_type == 'reduce':
            self.scheduler.step(eval_loss)
        else:
            self.scheduler.step()
        self.lr_info()

    def lr_info(self):
        last_lr = self.scheduler._last_lr[0]
        if last_lr != self.last_lr:
            self.logger.info(wrap_message(f'Updating learning rate ({self.last_lr} --> {last_lr}).', color='red'))
            self.last_lr = last_lr


class EarlyStopping:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.patience = args.patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = args.delta
        self.minimize = True if args.metric_mode == 'min' else False

    def __call__(self, score, model, path):
        if self.minimize:
            score = -score

        if self.best_score is None:
            self.save_checkpoint(score, model, path)
            self.best_score = score

        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.logger.info(wrap_message(f'EarlyStopping counter: {self.counter} out of {self.patience}', color='red'))
            self.early_stop = True if self.counter >= self.patience else False

        else:
            self.save_checkpoint(score, model, path)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, path):
        if self.minimize:
            last_score = -self.best_score if self.best_score is not None else np.inf
            self.logger.info(wrap_message(f'Evaluation score decreased ({last_score:.6f} --> {-score:.6f}).  Saving model ...', color='red'))
        else:
            last_score = self.best_score if self.best_score is not None else -np.inf
            self.logger.info(wrap_message(f'Evaluation score increased ({last_score:.6f} --> {score:.6f}).  Saving model ...', color='red'))

        save_path = os.path.join(path, "model.pt")
        torch.save(model.state_dict(), save_path)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    os.makedirs(os.basename(name), exist_ok=True)

    f = plt.figure(figsize=(4, 3), dpi=100)
    f.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
    palette = sns.color_palette('deep')
    sns.set_theme(
        style='darkgrid', context='paper', font='Arial', font_scale=1.6, 
        palette=[palette[0], palette[3], palette[2]] + palette[4:]
    )

    f.add_subplot(1, 1, 1)
    plt.plot(true, label='GroundTruth', linewidth=2, color='black', zorder=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color=palette[1], zorder=1)
    plt.legend(ncol=1, loc='lower right')
    plt.tick_params(labelsize=12)
    plt.tight_layout()

    with PdfPages(name) as pdf:
        pdf.savefig(f, bbox_inches='tight', pad_inches=0.01)


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class EvalAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            values = eval(values)
        except:
            try:
                values = eval(values.lower().capitalize())
            except:
                pass
        setattr(namespace, self.dest, values)


def get_shared_parameters(model, tag="shared"):
    if hasattr(model, "shared_parameters"):
        return model.shared_parameters()

    shared_parameters = []
    for name, param in model.named_parameters():
        if tag in name:
            shared_parameters.append(param)
    if len(shared_parameters) == 0:
        return model.parameters()
    return shared_parameters


def get_task_specific_parameters(model, tag="task_specific"):
    if hasattr(model, "task_specific_parameters"):
        return model.task_specific_parameters()

    task_specific_parameters = []
    for name, param in model.named_parameters():
        if tag in name:
            task_specific_parameters.append(param)
    if len(task_specific_parameters) == 0:
        return []
    return task_specific_parameters


def get_nb_trainable_parameters_info(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"\ntrainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}\n")


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class PParameter(nn.Parameter):
    def __repr__(self):
        tensor_type = str(self.data.type()).split('.')[-1]
        size_str = " x ".join(map(str, self.shape))
        return f"Parameter containing: [{tensor_type} of size {size_str}]"


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_yaml(data, yaml_file):
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f)


def update_dict(dic, key, value):
    if key not in dic:
        dic[key] = [value]
    elif key in dic:
        dic[key].append(value)
    return dic


def update_dict_multikeys(dic, new_dic):
    for key, value in new_dic.items():
        dic = update_dict(dic, key, value)
    return dic


class _ParameterDict(nn.ParameterDict):
    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = ', '.join(map(str, p.size()))
            device_str = '' if not p.is_cuda else f', device=GPU:{p.get_device()}'
            grad_str = '' if p.requires_grad else ', requires_grad=False'
            parastr = f'Parameter({size_str}, dtype={torch.typename(p)}){grad_str}{device_str}'
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr
