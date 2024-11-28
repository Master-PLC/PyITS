import os
import shutil

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils.tools import ensure_path


class Base_Estimator:
    def __init__(self, args, dataset, model, device=torch.device('cpu'), logger=None):
        self.args = args

        self.task_name = args.task_name
        self.save_dir = args.save_dir
        self.dataset = dataset
        self.device = device
        self.model = model

        self.logger = logger
        self.writer = self._create_writer()

        self.epoch = 0
        self.step = 0

        self.metric_train = {}
        self.metric_eval = {}
        self.metric_test = {}

        self.output_pred = args.output_pred
        self.output_vis = args.output_vis

    def _create_writer(self):
        item_list = os.listdir(self.save_dir)
        item_path_list = [os.path.join(self.save_dir, item) for item in item_list]
        item_path_list = [item_path for item_path in item_path_list if 'events' in item_path]
        if len(item_path_list) > 0:
            pre_log_dir = os.path.join(self.save_dir, "pre_logs")
            ensure_path(pre_log_dir)

            item_list = [os.path.basename(item_path) for item_path in item_path_list]
            for item, item_path in zip(item_list, item_path_list):
                shutil.move(item_path, os.path.join(pre_log_dir, item))

        return SummaryWriter(self.save_dir)

    def eval(self):
        pass

    def fit(self):
        pass

    def test(self):
        pass

    def record(self):
        with open(self.save_dir+'/train.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.metric_train, f, encoding='utf-8', allow_unicode=True)
        with open(self.save_dir+'/eval.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.metric_eval, f, encoding='utf-8', allow_unicode=True)
        with open(self.save_dir+'/test.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.metric_test, f, encoding='utf-8', allow_unicode=True)
        with open(self.save_dir+'/config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(vars(self.args), f, encoding='utf-8', allow_unicode=True)
