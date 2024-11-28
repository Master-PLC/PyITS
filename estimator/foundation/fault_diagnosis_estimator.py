import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from estimator.base_estimator import Base_Estimator
from utils.metrics import metric_collector
from utils.tools import EarlyStopping, Scheduler, update_dict_multikeys

warnings.filterwarnings('ignore')


class Fault_Diagnosis_Estimator(Base_Estimator):
    def __init__(self, args, dataset, model, device=torch.device('cpu'), logger=None):
        super().__init__(args, dataset, model, device, logger)

        assert self.task_name == 'fault_diagnosis'

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_scheduler(self, model_optim):
        scheduler = Scheduler(self.args, model_optim, logger=self.logger)
        return scheduler

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def eval(self, x, y, criterion):
        self.model.eval()

        eval_steps = len(x) // self.args.batch_size
        index = np.arange(len(x))
        idx = -1

        preds, trues = [], []
        eval_time = time.time()
        for idx in range(eval_steps):
            batch_index = index[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
            batch_x = torch.tensor(x[batch_index], device=self.device)  # [B, L, D]
            label = torch.tensor(y[batch_index], device=self.device)  # [B, 1]

            # encoder - decoder
            if self.args.output_attention:
                outputs, attns = self.model(batch_x, None, None, None)  # [B, C]
            else:
                outputs = self.model(batch_x, None, None, None)  # [B, C]

            preds.append(outputs.detach().cpu())
            trues.append(label.long().cpu())

        if len(x) % self.args.batch_size != 0:
            batch_index = index[(idx + 1) * self.args.batch_size:]
            batch_x = torch.tensor(x[batch_index], device=self.device)  # [B, L, D]
            label = torch.tensor(y[batch_index], device=self.device)  # [B, 1]

            # encoder - decoder
            if self.args.output_attention:
                outputs, attns = self.model(batch_x, None, None, None)  # [B, C]
            else:
                outputs = self.model(batch_x, None, None, None)  # [B, C]

            preds.append(outputs.detach().cpu())
            trues.append(label.long().cpu())

        preds = torch.cat(preds, dim=0)  # [B, C]
        trues = torch.cat(trues, dim=0)  # [B]
        total_loss = criterion(preds, trues.squeeze()).item()

        probs = F.softmax(preds, dim=1)  # [B, C]
        predictions = torch.argmax(probs, dim=1)  # [B]

        eval_metrics = metric_collector(
            predictions.numpy(), trues.numpy(), task_name=self.task_name, probs=probs.numpy()
        )
        update_dict_multikeys(self.metric_eval, eval_metrics)

        self.logger.info(f'Evaluation cost time: {time.time()-eval_time}')
        self.model.train()
        return total_loss

    def fit(self):
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        early_stopping = EarlyStopping(self.args, logger=self.logger)
        criterion = self._select_criterion()

        x, y = self.dataset.get_data(flag='train')
        x_eval, y_eval = self.dataset.get_data(flag='eval')
        train_steps = len(x) // self.args.batch_size  # drop last
        index = np.arange(len(x))

        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            np.random.shuffle(index)  # shuffle data
            epoch_time = time.time()
            for idx in range(train_steps):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                batch_index = index[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
                batch_x = torch.tensor(x[batch_index], device=self.device)  # [B, L, D]
                label = torch.tensor(y[batch_index], device=self.device)  # [B, 1]

                # encoder - decoder
                if self.args.output_attention:
                    outputs, attns = self.model(batch_x, None, None, None)  # [B, C]
                else:
                    outputs = self.model(batch_x, None, None, None)  # [B, C]
                    attns = None

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, label.long().squeeze())
                    loss += self.args.rec_lambda * loss_rec
                    self.writer.add_scalar(f'{self.task_name}/train/loss_rec', loss_rec, self.step)

                if isinstance(attns, torch.Tensor) and attns.ndim == 0:
                    # attns represents a scaler loss
                    loss += attns

                train_loss.append(loss.item())

                if (idx + 1) % 100 == 0:
                    self.logger.info(f"\titers: {idx+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - idx)
                    self.logger.info(f'\tspeed: {speed:.4f}s/iter; cost time: {cost_time:.4f}s; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                model_optim.step()

            self.logger.info(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time}")

            train_loss = np.average(train_loss)
            update_dict_multikeys(self.metric_train, {'loss': train_loss.item()})

            eval_loss = self.eval(x_eval, y_eval, criterion)
            self.writer.add_scalar(f'{self.task_name}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.task_name}/eval/loss', eval_loss, self.epoch)
            self.logger.info(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Eval Loss: {eval_loss:.7f}")

            early_stopping(eval_loss, self.model, self.save_dir)
            if early_stopping.early_stop:
                self.logger.info("Early stopping", color='red')
                break
            scheduler.step(epoch, eval_loss)

        best_model_path = os.path.join(self.save_dir, 'model.pt')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test=0):
        x, y = self.dataset.get_data(flag='test')
        if test:
            self.logger.info('loading pretrain model')
            pretrain_model_path = self.args.pretrain_model_path
            if pretrain_model_path and os.path.exists(pretrain_model_path):
                state_dict = torch.load(pretrain_model_path)
            else:
                state_dict = torch.load(os.path.join(self.save_dir, 'model.pt'))
            self.model.load_state_dict(state_dict)
            self.logger.info('loading pretrain model successfully')

        self.model.eval()

        test_steps = len(x) // self.args.test_batch_size
        index = np.arange(len(x))
        idx = -1

        preds, trues = [], []
        for idx in range(test_steps):
            batch_index = index[idx * self.args.test_batch_size: (idx + 1) * self.args.test_batch_size]
            batch_x = torch.tensor(x[batch_index], device=self.device)  # [B, L, D]
            label = torch.tensor(y[batch_index], device=self.device)  # [B, 1]

            # encoder - decoder
            if self.args.output_attention:
                outputs, attns = self.model(batch_x, None, None, None)  # [B, C]
            else:
                outputs = self.model(batch_x, None, None, None)  # [B, C]

            preds.append(outputs.detach().cpu())
            trues.append(label.long().cpu())

        if len(x) % self.args.test_batch_size != 0:
            batch_x = torch.tensor(x[index[(idx + 1) * self.args.test_batch_size:]], device=self.device)  # [B, L, D]
            label = torch.tensor(y[index[(idx + 1) * self.args.test_batch_size:]], device=self.device)  # [B, 1]

            # encoder - decoder
            if self.args.output_attention:
                outputs, attns = self.model(batch_x, None, None, None)  # [B, C]
            else:
                outputs = self.model(batch_x, None, None, None)  # [B, C]

            preds.append(outputs.detach().cpu())
            trues.append(label.long().cpu())

        preds = torch.cat(preds, dim=0)  # [B, C]
        probs = F.softmax(preds, dim=1)  # [B, C]
        predictions = torch.argmax(probs, dim=1).numpy()  # [B]

        probs = probs.numpy()  # [B, C]
        trues = torch.cat(trues, dim=0).squeeze().numpy()  # [B]

        self.logger.info(f'test shape: {predictions.shape}, {trues.shape}')
        self.metric_test = metric_collector(predictions, trues, task_name=self.task_name, probs=probs)

        if self.output_pred:
            np.save(os.path.join(self.save_dir, 'probs.npy'), probs)
            np.save(os.path.join(self.save_dir, 'pred.npy'), predictions)
            np.save(os.path.join(self.save_dir, 'true.npy'), trues)

        message = f"Fault Diagnosis Test | acc: {self.metric_test['accuracy']:.7f}, recall: {self.metric_test['recall']:.7f}"
        self.logger.info(message, color='red')

        for key, value in self.metric_test.items():
            self.writer.add_scalar(f'{self.task_name}/test/{key}', value, self.epoch)
        self.writer.close()

        self.record()
        return
