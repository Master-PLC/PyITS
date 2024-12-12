import os
import pickle as pkl
import time
import warnings

import numpy as np
import torch

from estimator.base_estimator import Base_Estimator
from utils.metrics import MSE, metric_collector
from utils.tools import update_dict_multikeys

warnings.filterwarnings('ignore')


class ML_Process_Monitoring_Estimator(Base_Estimator):
    def __init__(self, args, dataset, model, device=torch.device('cpu'), logger=None):
        super().__init__(args, dataset, model, device, logger)

        assert 'ml' in self.task_name
        base_task_name = self.task_name.split('_')[1:]
        base_task_name = [name.capitalize() for name in base_task_name]
        self.base_task_name = ' '.join(base_task_name)

    def _select_criterion(self):
        criterion = MSE
        return criterion

    def _save_checkpoint(self):
        save_path = os.path.join(self.save_dir, "model.pkl")
        with open(save_path, 'wb') as f:
            pkl.dump(self.model, f)

    def _load_checkpoint(self, path=None):
        save_path = os.path.join(self.save_dir, "model.pkl") if path is None else path
        with open(save_path, 'rb') as f:
            self.model = pkl.load(f)

    @staticmethod
    def _check_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return x

    def eval(self, x, y):
        eval_time = time.time()

        preds = self.model.predict(x)
        trues = y

        preds = preds[:, -1:]
        trues = trues[:, -1:]

        preds = self._check_numpy(preds)
        trues = self._check_numpy(trues)

        total_loss = MSE(preds, trues).item()
        eval_metrics = metric_collector(preds, trues, task_name=self.task_name)
        update_dict_multikeys(self.metric_eval, eval_metrics)

        self.logger.info(f'Evaluation cost time: {time.time()-eval_time}')
        return total_loss

    def fit(self):
        x, y = self.dataset.get_data(flag='train')
        x_eval, y_eval = self.dataset.get_data(flag='eval')
        index = np.arange(len(x))

        np.random.shuffle(index)  # shuffle data
        start_time = time.time()

        x = x[index]
        y = y[index]

        self.model.fit(x, y)
        self.logger.info(f"Fitting cost time: {time.time()-start_time}")

        train_loss = self.eval(x, y)
        eval_loss = self.eval(x_eval, y_eval)
        self.logger.info(f"Train Loss: {train_loss:.7f} Eval Loss: {eval_loss:.7f}")

        self._save_checkpoint()
        self._load_checkpoint()
        return self.model

    def test(self, test=0):
        x, y = self.dataset.get_data(flag='test')
        if test:
            self.logger.info('loading pretrain model')
            pretrain_model_path = self.args.pretrain_model_path
            if pretrain_model_path and os.path.exists(pretrain_model_path):
                self._load_checkpoint(pretrain_model_path)
            else:
                self._load_checkpoint()
            self.logger.info('loading pretrain model successfully')

        preds = self.model.predict(x)
        trues = y

        preds = preds[:, -1:]
        trues = trues[:, -1:]

        preds = self._check_numpy(preds)
        trues = self._check_numpy(trues)

        self.logger.info(f'test shape: {preds.shape}, {trues.shape}')
        self.metric_test = metric_collector(
            self._check_numpy(preds), self._check_numpy(trues), task_name=self.task_name
        )

        if self.output_pred:
            np.save(os.path.join(self.save_dir, 'input.npy'), x)
            np.save(os.path.join(self.save_dir, 'pred.npy'), preds)
            np.save(os.path.join(self.save_dir, 'true.npy'), trues)

        message = f"{self.base_task_name} Test | mse: {self.metric_test['mse']:.7f}, r2: {self.metric_test['r2']:.7f}"
        self.logger.info(message, color='red')

        for key, value in self.metric_test.items():
            self.writer.add_scalar(f'{self.task_name}/test/{key}', value, self.epoch)
        self.writer.close()

        self.record()
        return
