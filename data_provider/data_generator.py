import logging
import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_provider.data_utils import shift_data

warnings.filterwarnings('ignore')


class Base_Dataset:
    def __init__(self, configs, logger):
        self.configs = configs

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.features = configs.features
        self.target_idx = configs.target_idx
        self.scale = configs.scale
        self.timeenc = 0 if configs.embed != 'timeF' else 1
        self.freq = configs.freq
        self.data_percentage = configs.data_percentage
        self.shift = configs.shift

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.root_path = configs.root_path
        self.data_path = configs.data_path
        self.file_path = os.path.join(self.root_path, self.data_path)

        self.train_data = None
        self.eval_data = None
        self.test_data = None

    def generate_soft_sensor_data(self, **kwargs):
        raise NotImplementedError("This dataset does not support soft sensor data generation.")

    def generate_process_monitoring_data(self, **kwargs):
        raise NotImplementedError("This dataset does not support process monitoring data generation.")

    def generate_fault_diagnosis_data(self, **kwargs):
        raise NotImplementedError("This dataset does not support fault diagnosis data generation.")

    def generate_rul_estimation_data(self, **kwargs):
        raise NotImplementedError("This dataset does not support RUL estimation data generation.")

    def generate_predictive_maintenance_data(self, **kwargs):
        raise NotImplementedError("This dataset does not support predictive maintenance data generation.")

    def generate_data(self, task_name):
        if task_name == 'soft_sensor':
            self.generate_soft_sensor_data()
        elif task_name == 'soft_sensor_ml':
            self.generate_soft_sensor_data(flatten=True)
        elif task_name == 'process_monitoring':
            self.generate_process_monitoring_data()
        elif task_name == 'process_monitoring_ml':
            self.generate_process_monitoring_data(flatten=True)
        elif task_name == 'fault_diagnosis':
            self.generate_fault_diagnosis_data()
        elif task_name == 'rul_estimation':
            self.generate_rul_estimation_data()
        elif task_name == 'rul_estimation_ml':
            self.generate_rul_estimation_data(flatten=True)
        elif task_name == 'predictive_maintenance':
            self.generate_predictive_maintenance_data()
        else:
            raise ValueError("Invalid task name for generating data.")
        return self.configs

    def get_data(self, flag='train'):
        if flag == 'train':
            return self.train_data
        elif flag == 'eval':
            return self.eval_data
        elif flag == 'test':
            return self.test_data
        else:
            raise ValueError("Invalid flag for getting data.")


class Dataset_SRU(Base_Dataset):
    """Dataset source: https://www.sciencedirect.com/science/article/pii/S0967066103000790
    """
    def __init__(self, configs, logger):
        super().__init__(configs, logger)

        self.xcols = ['u1', 'u2', 'u3', 'u4', 'u5']
        self.ycols = ['y1', 'y2']
        self.cols = ['u1', 'u2', 'u3', 'u4', 'u5', 'y1', 'y2']
        self.configs.freq = 'm'  # 1 minute

    def generate_soft_sensor_data(self, flatten=False):
        df_raw = pd.read_csv(
            self.file_path, header=None, skiprows=1, skip_blank_lines=True, sep='  ', 
            engine='python', names=self.cols
        )
        ycols = [self.ycols[i] for i in self.target_idx]
        df_raw = df_raw[self.xcols + ycols]

        if self.shift > 0:
            df_raw, shifted_cols = shift_data(df_raw, columns=ycols, shift=self.shift)
            df_raw = df_raw[self.xcols + shifted_cols + ycols]

        self.logger.info('Head lines of raw dataframe:')
        self.logger.info(df_raw.head())

        dataset = df_raw.to_numpy().astype(np.float32)
        data_train, data_eval, data_test = dataset[:int(len(dataset) * 0.7)], dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.85)], dataset[int(len(dataset) * 0.85):]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train)
            data_train, data_eval, data_test = self.scaler.transform(data_train), self.scaler.transform(data_eval), self.scaler.transform(data_test)

        data_train = np.lib.stride_tricks.sliding_window_view(data_train, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        data_eval = np.lib.stride_tricks.sliding_window_view(data_eval, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        data_test = np.lib.stride_tricks.sliding_window_view(data_test, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))

        self.train_data = [data_train[..., :-len(ycols)], data_train[:, -1, -len(ycols):]]  # [N, L, Dx], [N, Dy]
        self.eval_data = [data_eval[..., :-len(ycols)], data_eval[:, -1, -len(ycols):]]
        self.test_data = [data_test[..., :-len(ycols)], data_test[:, -1, -len(ycols):]]

        if flatten:
            self.train_data[0] = self.train_data[0].reshape(self.train_data[0].shape[0], -1)  # [N, L*Dx]
            self.eval_data[0] = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1)
            self.test_data[0] = self.test_data[0].reshape(self.test_data[0].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]

    def generate_process_monitoring_data(self, flatten=False):
        df_raw = pd.read_csv(
            self.file_path, header=None, skiprows=1, skip_blank_lines=True, sep='  ', 
            engine='python', names=self.cols
        )
        ycols = [self.cols[i] for i in self.target_idx]
        df_raw = df_raw[ycols]

        if self.shift > 0:
            df_raw, shifted_cols = shift_data(df_raw, columns=ycols, shift=self.shift)
            df_raw = df_raw[shifted_cols + ycols]

        self.logger.info('Head lines of raw dataframe:')
        self.logger.info(df_raw.head())

        dataset = df_raw.to_numpy().astype(np.float32)
        data_train, data_eval, data_test = dataset[:int(len(dataset) * 0.7)], dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.85)], dataset[int(len(dataset) * 0.85):]
        
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train)
            data_train, data_eval, data_test = self.scaler.transform(data_train), self.scaler.transform(data_eval), self.scaler.transform(data_test)

        data_train = np.lib.stride_tricks.sliding_window_view(data_train, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        data_eval = np.lib.stride_tricks.sliding_window_view(data_eval, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        data_test = np.lib.stride_tricks.sliding_window_view(data_test, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))

        self.train_data = [data_train[:, :self.seq_len, :], data_train[:, -(self.label_len+self.pred_len):, -len(ycols):]]  # [N, L, Dx], [N, S+P, Dy]
        self.eval_data = [data_eval[:, :self.seq_len, :], data_eval[:, -(self.label_len+self.pred_len):, -len(ycols):]]
        self.test_data = [data_test[:, :self.seq_len, :], data_test[:, -(self.label_len+self.pred_len):, -len(ycols):]]

        if flatten:
            self.train_data = self.train_data[0].reshape(self.train_data[0].shape[0], -1), self.train_data[1].reshape(self.train_data[1].shape[0], -1)  # [N, L*Dx], [N, (S+P)*Dy]
            self.eval_data = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1), self.eval_data[1].reshape(self.eval_data[1].shape[0], -1)
            self.test_data = self.test_data[0].reshape(self.test_data[0].shape[0], -1), self.test_data[1].reshape(self.test_data[1].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]


class Dataset_Debutanizer(Dataset_SRU):
    """Dataset source: https://github.com/josmellcordova/Debutanizer
    Paper link: https://link.springer.com/book/10.1007/978-1-84628-480-9
    """
    def __init__(self, configs, logger):
        super().__init__(configs, logger)

        self.xcols = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7']
        self.ycols = ['y']
        self.cols = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y']
        self.configs.freq = 'm'  # 6 minute


class Dataset_TEP(Base_Dataset):
    """Dataset source: http://web.mit.edu/braatzgroup/TE_process.zip
    """
    def __init__(self, configs, logger):
        super().__init__(configs, logger)

        self.n_trial = 22
        self.xcols = [f'XMEAS{i+1}' for i in range(41)]
        self.ycols = [f'XMV{i+1}' for i in range(11)]
        self.cols = [f'XMEAS{i+1}' for i in range(41)] + [f'XMV{i+1}' for i in range(11)]
        self.first_col = 'Trial'
        self.last_col = 'Fault'
        self.configs.freq = 'm'  # 3 minute

    def generate_soft_sensor_data(self, flatten=False):
        train = pd.read_csv(os.path.join(self.root_path, 'train.csv'), header=0)
        test = pd.read_csv(os.path.join(self.root_path, 'test.csv'), header=0)

        ycols = [self.ycols[i] for i in self.target_idx]
        train = train[['Trial'] + self.xcols + ycols]
        test = test[['Trial'] + self.xcols + ycols]

        if self.shift > 0:
            train, shifted_cols = shift_data(train, columns=ycols, shift=self.shift)
            test, _ = shift_data(test, columns=ycols, shift=self.shift)

            train = train[['Trial'] + self.xcols + shifted_cols + ycols]
            test = test[['Trial'] + self.xcols + shifted_cols + ycols]

        self.logger.info('Head lines of raw dataframe:')
        self.logger.info(train.head())

        data_train = train.groupby('Trial').apply(lambda x: x.iloc[:int(len(x) * 0.8)]).reset_index(drop=True)
        data_eval = train.groupby('Trial').apply(lambda x: x.iloc[int(len(x) * 0.8):]).reset_index(drop=True)
        data_test = test

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train.iloc[:, 1:])
            data_train.iloc[:, 1:] = self.scaler.transform(data_train.iloc[:, 1:])
            data_eval.iloc[:, 1:] = self.scaler.transform(data_eval.iloc[:, 1:])
            data_test.iloc[:, 1:] = self.scaler.transform(data_test.iloc[:, 1:])

        data_train = data_train.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values
        data_eval = data_eval.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values
        data_test = data_test.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values

        data_train = np.concatenate(data_train, axis=0).astype(np.float32)
        data_eval = np.concatenate(data_eval, axis=0).astype(np.float32)
        data_test = np.concatenate(data_test, axis=0).astype(np.float32)

        self.train_data = [data_train[..., :-len(ycols)], data_train[:, -1, -len(ycols):]]  # [N, L, Dx], [N, Dy]
        self.eval_data = [data_eval[..., :-len(ycols)], data_eval[:, -1, -len(ycols):]]
        self.test_data = [data_test[..., :-len(ycols)], data_test[:, -1, -len(ycols):]]
        
        if flatten:
            self.train_data[0] = self.train_data[0].reshape(self.train_data[0].shape[0], -1)  # [N, L*Dx]
            self.eval_data[0] = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1)
            self.test_data[0] = self.test_data[0].reshape(self.test_data[0].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]

    def generate_process_monitoring_data(self, flatten=False):
        train = pd.read_csv(os.path.join(self.root_path, 'train.csv'), header=0)
        test = pd.read_csv(os.path.join(self.root_path, 'test.csv'), header=0)

        ycols = [self.cols[i] for i in self.target_idx]
        train = train[['Trial'] + ycols]
        test = test[['Trial'] + ycols]

        if self.shift > 0:
            train, shifted_cols = shift_data(train, columns=ycols, shift=self.shift)
            test, _ = shift_data(test, columns=ycols, shift=self.shift)

            train = train[['Trial'] + shifted_cols + ycols]
            test = test[['Trial'] + shifted_cols + ycols]

        self.logger.info('Head lines of raw dataframe:')
        self.logger.info(train.head())

        data_train = train.groupby('Trial').apply(lambda x: x.iloc[:int(len(x) * 0.8)]).reset_index(drop=True)
        data_eval = train.groupby('Trial').apply(lambda x: x.iloc[int(len(x) * 0.8):]).reset_index(drop=True)
        data_test = test

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train.iloc[:, 1:])
            data_train.iloc[:, 1:] = self.scaler.transform(data_train.iloc[:, 1:])
            data_eval.iloc[:, 1:] = self.scaler.transform(data_eval.iloc[:, 1:])
            data_test.iloc[:, 1:] = self.scaler.transform(data_test.iloc[:, 1:])

        data_train = data_train.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        ).values
        data_eval = data_eval.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        ).values
        data_test = data_test.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        ).values

        data_train = np.concatenate(data_train, axis=0).astype(np.float32)
        data_eval = np.concatenate(data_eval, axis=0).astype(np.float32)
        data_test = np.concatenate(data_test, axis=0).astype(np.float32)

        self.train_data = [data_train[:, :self.seq_len, :], data_train[:, -(self.label_len+self.pred_len):, -len(ycols):]]  # [N, L, Dx], [N, S+P, Dy]
        self.eval_data = [data_eval[:, :self.seq_len, :], data_eval[:, -(self.label_len+self.pred_len):, -len(ycols):]]
        self.test_data = [data_test[:, :self.seq_len, :], data_test[:, -(self.label_len+self.pred_len):, -len(ycols):]]

        if flatten:
            self.train_data = self.train_data[0].reshape(self.train_data[0].shape[0], -1), self.train_data[1].reshape(self.train_data[1].shape[0], -1)  # [N, L*Dx], [N, (S+P)*Dy]
            self.eval_data = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1), self.eval_data[1].reshape(self.eval_data[1].shape[0], -1)
            self.test_data = self.test_data[0].reshape(self.test_data[0].shape[0], -1), self.test_data[1].reshape(self.test_data[1].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]

    def generate_fault_diagnosis_data(self, flatten=False):
        train = pd.read_csv(os.path.join(self.root_path, 'train.csv'), header=0)
        test = pd.read_csv(os.path.join(self.root_path, 'test.csv'), header=0)

        train_data = train[['Trial'] + self.cols]
        train_labels = train['Fault'].astype('category')
        self.class_names = train_labels.cat.categories
        train_labels = pd.DataFrame(train_labels.cat.codes, dtype=np.int8)
        train = pd.concat([train_data, train_labels], axis=1)

        test_data = test[['Trial'] + self.cols]
        test_labels = test['Fault'].astype('category')
        test_labels.cat.set_categories(self.class_names)
        test_labels = pd.DataFrame(test_labels.cat.codes, dtype=np.int8)
        test = pd.concat([test_data, test_labels], axis=1)

        self.logger.info('Head lines of raw dataframe:')
        self.logger.info(train.head())

        data_train = train.groupby('Trial').apply(lambda x: x.iloc[:int(len(x) * 0.8)]).reset_index(drop=True)
        data_eval = train.groupby('Trial').apply(lambda x: x.iloc[int(len(x) * 0.8):]).reset_index(drop=True)
        data_test = test

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train.iloc[:, 1:-1])
            data_train.iloc[:, 1:-1] = self.scaler.transform(data_train.iloc[:, 1:-1])
            data_eval.iloc[:, 1:-1] = self.scaler.transform(data_eval.iloc[:, 1:-1])
            data_test.iloc[:, 1:-1] = self.scaler.transform(data_test.iloc[:, 1:-1])

        data_train = data_train.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values
        data_eval = data_eval.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values
        data_test = data_test.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values

        data_train = np.concatenate(data_train, axis=0).astype(np.float32)
        data_eval = np.concatenate(data_eval, axis=0).astype(np.float32)
        data_test = np.concatenate(data_test, axis=0).astype(np.float32)

        self.train_data = [data_train[..., :-1], data_train[:, -1, -1:]]  # [N, L, Dx], [N, 1]
        self.eval_data = [data_eval[..., :-1], data_eval[:, -1, -1:]]
        self.test_data = [data_test[..., :-1], data_test[:, -1, -1:]]

        if flatten:
            self.train_data[0] = self.train_data[0].reshape(self.train_data[0].shape[0], -1)
            self.eval_data[0] = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1)
            self.test_data[0] = self.test_data[0].reshape(self.test_data[0].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.c_out = len(self.class_names)


class Dataset_CWRU(Base_Dataset):
    def __init__(self, configs, logger):
        super().__init__(configs, logger)

        self.feature_names = ['DE', 'FE']
        self.label_name = 'fault'
        self.configs.freq = 's'

    def generate_fault_diagnosis_data(self, flatten=False):
        with open(self.file_path, 'rb') as f:
            df_raw = pkl.load(f)

        df_data = df_raw[self.feature_names]
        if self.data_percentage < 1.:
            self.logger.info(f"Shrink the train data to {self.data_percentage * 100}%")
            df_data = df_data.applymap(lambda x: x[int(len(x) * (1 - self.data_percentage)):])
        labels = df_raw[self.label_name].astype('category')
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        data_train = df_data.applymap(lambda x: x[:int(len(x) * 0.7)]).values
        data_train = [np.stack(x, axis=1).astype(np.float32) for x in data_train]
        data_eval = df_data.applymap(lambda x: x[int(len(x) * 0.7):int(len(x) * 0.85)]).values
        data_eval = [np.stack(x, axis=1).astype(np.float32) for x in data_eval]
        data_test = df_data.applymap(lambda x: x[int(len(x) * 0.85):]).values
        data_test = [np.stack(x, axis=1).astype(np.float32) for x in data_test]

        if self.scale:
            self.scaler = StandardScaler()
            train_data = np.concatenate(data_train, axis=0)
            self.scaler.fit(train_data)

            data_train = [self.scaler.transform(x) for x in data_train]
            data_eval = [self.scaler.transform(x) for x in data_eval]
            data_test = [self.scaler.transform(x) for x in data_test]

        data_train = [np.lib.stride_tricks.sliding_window_view(x, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1)) for x in data_train]
        data_eval = [np.lib.stride_tricks.sliding_window_view(x, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1)) for x in data_eval]
        data_test = [np.lib.stride_tricks.sliding_window_view(x, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1)) for x in data_test]

        train_labels = [lbl for label, data in zip(labels_df.values, data_train) for lbl in [label] * data.shape[0]]
        eval_labels = [lbl for label, data in zip(labels_df.values, data_eval) for lbl in [label] * data.shape[0]]
        test_labels = [lbl for label, data in zip(labels_df.values, data_test) for lbl in [label] * data.shape[0]]

        self.train_data = [np.concatenate(data_train, axis=0), np.array(train_labels).reshape(-1, 1)]  # [N, L, D], [N, 1]
        self.eval_data = [np.concatenate(data_eval, axis=0), np.array(eval_labels).reshape(-1, 1)]
        self.test_data = [np.concatenate(data_test, axis=0), np.array(test_labels).reshape(-1, 1)]

        if flatten:
            self.train_data[0] = self.train_data[0].reshape(self.train_data[0].shape[0], -1)
            self.eval_data[0] = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1)
            self.test_data[0] = self.test_data[0].reshape(self.test_data[0].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.c_out = len(self.class_names)


class Dataset_C_MAPSS(Base_Dataset):
    def __init__(self, configs, logger):
        super().__init__(configs, logger)

        assert self.data_path in ['FD001', 'FD002', 'FD003', 'FD004'], "Invalid sub-dataset name for C-MAPSS dataset."
        self.cols = ['unit_id', 'time_cycles'] + [f's_{i+1}' for i in range(21)] + ['RUL', 'op_cond']
        self.xcols = [f's_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
        self.threshold = 125 if self.data_path in ['FD001', 'FD003'] else 130
        self.configs.freq = 's'

    def generate_rul_estimation_data(self, flatten=False):
        X_train = pd.read_csv(os.path.join(self.root_path, f"X_train_{self.data_path}.csv"), header=0)
        X_train['RUL'].clip(upper=self.threshold, inplace=True)
        X_train['RUL'] = X_train['RUL'].apply(lambda x: x / self.threshold)

        X_test = pd.read_csv(os.path.join(self.root_path, f"X_test_{self.data_path}.csv"), header=0)
        y_test = pd.read_csv(os.path.join(self.root_path, f"y_test_{self.data_path}.csv"), header=0)
        y_test['RUL'].clip(upper=self.threshold, inplace=True)
        y_test['RUL'] = y_test['RUL'].apply(lambda x: x / self.threshold)

        unique_units = X_train['unit_id'].unique()
        unit_train = unique_units[:int(len(unique_units) * 0.8)]
        unit_eval = unique_units[int(len(unique_units) * 0.8):]
        conditions = X_train['op_cond'].unique()

        data_train = X_train[X_train['unit_id'].isin(unit_train)]
        data_eval = X_train[X_train['unit_id'].isin(unit_eval)]
        data_test = X_test

        if self.scale:
            self.scalers = {cond: StandardScaler() for cond in conditions}
            for condition in conditions:
                self.scalers[condition].fit(data_train.loc[data_train['op_cond'] == condition, self.xcols])

            for condition in data_train['op_cond'].unique():
                data_train.loc[data_train['op_cond'] == condition, self.xcols] = self.scalers[condition].transform(data_train.loc[data_train['op_cond'] == condition, self.xcols])
            for condition in data_eval['op_cond'].unique():
                data_eval.loc[data_eval['op_cond'] == condition, self.xcols] = self.scalers[condition].transform(data_eval.loc[data_eval['op_cond'] == condition, self.xcols])
            for condition in data_test['op_cond'].unique():
                data_test.loc[data_test['op_cond'] == condition, self.xcols] = self.scalers[condition].transform(data_test.loc[data_test['op_cond'] == condition, self.xcols])

        train_data = []
        for unit_id, df_unit in data_train.groupby('unit_id'):
            data = df_unit[self.xcols + ['RUL']].values.astype(np.float32)
            train_data.append(np.lib.stride_tricks.sliding_window_view(data, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1)))
        train_data = np.concatenate(train_data, axis=0)

        eval_data = []
        for unit_id, df_unit in data_eval.groupby('unit_id'):
            data = df_unit[self.xcols + ['RUL']].values.astype(np.float32)
            eval_data.append(np.lib.stride_tricks.sliding_window_view(data, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1)))
        eval_data = np.concatenate(eval_data, axis=0)

        test_data = []
        for unit_id, df_unit in data_test.groupby('unit_id'):
            data = df_unit[self.xcols].values.astype(np.float32)
            data = np.lib.stride_tricks.sliding_window_view(data, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
            test_data.append(data[-1:])
        test_data = np.concatenate(test_data, axis=0)

        self.train_data = [train_data[..., :-1], train_data[:, -1, -1:]]  # [N, L, D], [N, 1]
        self.eval_data = [eval_data[..., :-1], eval_data[:, -1, -1:]]
        self.test_data = [test_data, y_test['RUL'].values.reshape(-1, 1)]

        if flatten:
            self.train_data[0] = self.train_data[0].reshape(self.train_data[0].shape[0], -1)
            self.eval_data[0] = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1)
            self.test_data[0] = self.test_data[0].reshape(self.test_data[0].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]


class Dataset_NASA_Li_ion(Base_Dataset):
    def __init__(self, configs, logger):
        super().__init__(configs, logger)

        self.train_subsets = ['B0005', 'B0006', 'B0007', 'B0018']
        assert self.data_path in self.train_subsets, "Invalid battery name for NASA Li-ion dataset testing."
        self.train_subsets.remove(self.data_path)

        self.dtype = 'discharge'
        self.xcols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']
        self.cols = ['Battery', 'Type', 'Start_Time', 'Cycle', 'Ambient_Temp'] + \
            ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Capacity', 'Time']
        self.upper_threshold = 2.0
        self.lower_threshold = self.upper_threshold * (1 - 0.3)
        self.configs.freq = 's'

    def _process_subset_data(self, df_raw):
        df_raw = df_raw[self.dtype][self.xcols + ['Capacity']]
        df_raw['Capacity'].clip(lower=self.lower_threshold, upper=self.upper_threshold, inplace=True)
        df_raw['Capacity'] = df_raw['Capacity'].apply(lambda x: (x - self.lower_threshold) / (self.upper_threshold - self.lower_threshold))
        return df_raw

    def generate_rul_estimation_data(self, flatten=False):
        data_train = []
        for subset in self.train_subsets:
            with open(os.path.join(self.root_path, f"{subset}.pkl"), 'rb') as f:
                df_raw = pkl.load(f)
            df_raw = self._process_subset_data(df_raw)
            data_train.append(df_raw)

        with open(os.path.join(self.root_path, f"{self.data_path}.pkl"), 'rb') as f:
            df_raw = pkl.load(f)
        data_test = self._process_subset_data(df_raw)

        if self.scale:
            self.scaler = StandardScaler()
            data_train_ = pd.concat(data_train, axis=0).iloc[:, :-1]
            self.scaler.fit(data_train_)
            for i, df in enumerate(data_train):
                data_train[i].iloc[:, :-1] = self.scaler.transform(df.iloc[:, :-1])
            data_test.iloc[:, :-1] = self.scaler.transform(data_test.iloc[:, :-1])

        train_data = []
        for df in data_train:
            df = df.groupby('Capacity').apply(
                lambda x: np.lib.stride_tricks.sliding_window_view(x.values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
            ).values
            df = np.concatenate(df, axis=0)
            train_data.append(df)
        train_data = np.concatenate(train_data, axis=0).astype(np.float32)

        test_data = data_test.groupby('Capacity').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.values, window_shape=(self.seq_len), axis=0).transpose((0, 2, 1))
        ).values
        test_data = [x[-1:] for x in test_data]
        test_data = np.concatenate(test_data, axis=0).astype(np.float32)

        self.train_data = [train_data[..., :-1], train_data[:, -1, -1:]]  # [N, L, D], [N, 1]
        self.eval_data = [test_data[..., :-1], test_data[:, -1, -1:]]
        self.test_data = [test_data[..., :-1], test_data[:, -1, -1:]]

        if flatten:
            self.train_data[0] = self.train_data[0].reshape(self.train_data[0].shape[0], -1)
            self.eval_data[0] = self.eval_data[0].reshape(self.eval_data[0].shape[0], -1)
            self.test_data[0] = self.test_data[0].reshape(self.test_data[0].shape[0], -1)

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]


class Dataset_SWaT(Base_Dataset):
    def __init__(self, configs, logger):
        super().__init__(configs, logger)
        self.configs.freq = 's'

    def generate_predictive_maintenance_data(self, flatten=False):
        train_data = pd.read_csv(os.path.join(self.root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(self.root_path, 'swat2.csv'))

        if self.data_percentage < 1.:
            self.logger.info(f"Shrink the train data to {self.data_percentage * 100}%")
            skip_len = int(len(train_data) * (1 - self.data_percentage))
            train_data = train_data[skip_len:]

        num_train = int(len(train_data) * 0.8)

        data_train = train_data[:num_train].values
        data_eval = train_data[num_train:].values
        data_test = test_data.values

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train[:, :-1])

            data_train[:, :-1] = self.scaler.transform(data_train[:, :-1])
            data_eval[:, :-1] = self.scaler.transform(data_eval[:, :-1])
            data_test[:, :-1] = self.scaler.transform(data_test[:, :-1])

        data_train = np.lib.stride_tricks.sliding_window_view(data_train, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        data_eval = np.lib.stride_tricks.sliding_window_view(data_eval, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        data_test = np.lib.stride_tricks.sliding_window_view(data_test, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))

        self.train_data = [data_train[:, :self.seq_len, :-1].astype(np.float32), data_train[:, -self.pred_len:, -1:], data_train[:, -self.pred_len:, :-1].astype(np.float32)]  # [N, L, D], [N, P, 1], [N, P, D]
        self.eval_data = [data_eval[:, :self.seq_len, :-1].astype(np.float32), data_eval[:, -self.pred_len:, -1:], data_eval[:, -self.pred_len:, :-1].astype(np.float32)]
        self.test_data = [data_test[:, :self.seq_len, :-1].astype(np.float32), data_test[:, -self.pred_len:, -1:], data_test[:, -self.pred_len:, :-1].astype(np.float32)]

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]


class Dataset_SKAB(Base_Dataset):
    def __init__(self, configs, logger):
        super().__init__(configs, logger)
        self.cols = ['datetime', 'Trial'] + \
            ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS'] + \
            ['anomaly', 'changepoint']
        self.configs.freq = 's'

    def generate_predictive_maintenance_data(self, flatten=False):
        df_raw = pd.read_csv(self.file_path, header=0)
        df_raw = df_raw.iloc[:, 1:-1]  # Trial to anomaly

        data_train = df_raw.groupby('Trial').apply(lambda x: x.iloc[:int(len(x) * 0.7)]).reset_index(drop=True)
        data_eval = df_raw.groupby('Trial').apply(lambda x: x.iloc[int(len(x) * 0.7):int(len(x) * 0.85)]).reset_index(drop=True)
        data_test = df_raw.groupby('Trial').apply(lambda x: x.iloc[int(len(x) * 0.85):]).reset_index(drop=True)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_train.iloc[:, 1:-1])  # Exclude Trial and anomaly

            data_train.iloc[:, 1:-1] = self.scaler.transform(data_train.iloc[:, 1:-1])
            data_eval.iloc[:, 1:-1] = self.scaler.transform(data_eval.iloc[:, 1:-1])
            data_test.iloc[:, 1:-1] = self.scaler.transform(data_test.iloc[:, 1:-1])

        data_train = data_train.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))  # Exclude Trial
        ).values
        data_eval = data_eval.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        ).values
        data_test = data_test.groupby('Trial').apply(
            lambda x: np.lib.stride_tricks.sliding_window_view(x.iloc[:, 1:].values, window_shape=(self.seq_len+self.pred_len), axis=0).transpose((0, 2, 1))
        ).values

        data_train = np.concatenate(data_train, axis=0).astype(np.float32)
        data_eval = np.concatenate(data_eval, axis=0).astype(np.float32)
        data_test = np.concatenate(data_test, axis=0).astype(np.float32)

        self.train_data = [data_train[:, :self.seq_len, :-1].astype(np.float32), data_train[:, -self.pred_len:, -1:].astype(np.int8), data_train[:, -self.pred_len:, :-1].astype(np.float32)]  # [N, L, D], [N, P, 1], [N, P, D]
        self.eval_data = [data_eval[:, :self.seq_len, :-1].astype(np.float32), data_eval[:, -self.pred_len:, -1:].astype(np.int8), data_eval[:, -self.pred_len:, :-1].astype(np.float32)]
        self.test_data = [data_test[:, :self.seq_len, :-1].astype(np.float32), data_test[:, -self.pred_len:, -1:].astype(np.int8), data_test[:, -self.pred_len:, :-1].astype(np.float32)]

        self.configs.enc_in = self.train_data[0].shape[-1]
        self.configs.dec_in = self.train_data[1].shape[-1]
        self.configs.c_out = self.train_data[1].shape[-1]
