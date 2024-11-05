###################################################################################################
#
# Copyright (C) 2024 ITRVN. All Rights Reserved.
# This software is proprietary to ITRVN. and its licensors.
#
###################################################################################################
"""
Classes and functions for the ECG Heartbeat Categorization Dataset
https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data
"""
import os

import numpy as np
import torch
from torchvision import transforms

import git
import pandas as pd
from git.exc import GitCommandError

import ai8x
from utils.dataloader_utils import makedir_exist_ok

from .ecg_dataframe_parser import ECG_DataFrame_Parser


class SampleECG(ECG_DataFrame_Parser):
    """
    The PTB Diagnostic ECG Database
        Number of Samples: 14552
        Number of Categories: 2
        Sampling Frequency: 125Hz
        Data Source: Physionet's PTB Diagnostic Database

    Remark: All the samples are cropped, downsampled and padded with zeroes if necessary to the fixed dimension of 187.
    """

    train_ratio = 0.8

    def __init__(self, root, d_type,
                 transform,
                 eval_mode,
                 label_as_signal,
                 train_ratio=train_ratio,
                 accel_in_second_dim=True,
                 cnn_1dinput_len=187):

        self.root = root

        self.accel_in_second_dim = accel_in_second_dim

        self.processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        main_df = self.gen_dataframe()

        super().__init__(root,
                         d_type=d_type,
                         transform=transform,
                         eval_mode=eval_mode,
                         label_as_signal=label_as_signal,
                         train_ratio=train_ratio,
                         cnn_1dinput_len=cnn_1dinput_len,
                         main_df=main_df)



    def parse_ECG_and_return_common_df_row(self, file_full_path, label=None):
        """
        
        """
        raw_data = pd.read_csv(file_full_path, sep=',', header=None).iloc[:, :-1]
        raw_data = raw_data.to_numpy()

        return [os.path.basename(file_full_path).split('/')[-1], raw_data, label]

    def __getitem__(self, index):
        if self.accel_in_second_dim:
            signal, lbl = super().__getitem__(index)  # pylint: disable=unbalanced-tuple-unpacking
            signal = torch.transpose(signal, 0, 1)
            lbl = lbl.transpose()
            return signal, lbl

        return super().__getitem__(index)

    def gen_dataframe(self):
        """
        Generate dataframes from csv files of ECG
        """
        file_name = f'{self.__class__.__name__}_dataframe.pkl'
        df_path = \
            os.path.join(self.root, self.__class__.__name__, file_name)

        # if os.path.isfile(df_path):
        #     print(f'\nFile {file_name} already exists\n')
        #     main_df = pd.read_pickle(df_path)

        #     return main_df

        print('\nGenerating data frame pickle files from the raw data \n')

        
        data_dir = "/home/dattran/Project/MAX/dataset"

        if not os.path.isdir(data_dir):
            print(f'\nDataset directory {data_dir} does not exist.\n')
            return None

        with os.scandir(data_dir) as it:
            if not any(it):
                print(f'\nDataset directory {data_dir} is empty.\n')
                return None

        abnormal_data_list = []
        normal_data_list = []

        df_normals = self.create_common_empty_df()
        df_anormals = self.create_common_empty_df()

        for file in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, file)

            if file.endswith('_normal.csv'):
                normal_row = self.parse_ECG_and_return_common_df_row(file_full_path=full_path, label=0)
                normal_data_list.append(normal_row)

            else:
                abnormal_row = self.parse_ECG_and_return_common_df_row(file_full_path=full_path, label=1)
                abnormal_data_list.append(abnormal_row)

        df_normals = pd.DataFrame(data=np.array(normal_data_list, dtype=object),
                                  columns=self.common_dataframe_columns)

        df_anormals = pd.DataFrame(data=np.array(abnormal_data_list, dtype=object),
                                   columns=self.common_dataframe_columns)

        main_df = pd.concat([df_normals, df_anormals], axis=0)

        makedir_exist_ok(self.processed_folder)
        main_df.to_pickle(df_path)

        return main_df


def sampleecg_get_datasets(data, 
                           load_train=True, 
                           load_test=True,
                           eval_mode=False,
                           label_as_signal=True,
                           accel_in_second_dim=True,
                           cnn_1dinput_len=187):
    """
    Returns Sample ECG Dataset
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        train_dataset = SampleECG(root=data_dir, 
                                  d_type='train',
                                  transform=train_transform,
                                  eval_mode=eval_mode,
                                  label_as_signal=label_as_signal,
                                  accel_in_second_dim=accel_in_second_dim,
                                  cnn_1dinput_len=cnn_1dinput_len)

        print(f'Train dataset length: {len(train_dataset)}\n')
        # print(f'Train signal shape: {train_dataset.__getitem__(0)[0].shape}\n')
        # print(f'Train label shape: {train_dataset.__getitem__(0)[1].shape}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        test_dataset = SampleECG(root=data_dir, 
                                 d_type='test',
                                 transform=test_transform,
                                 eval_mode=eval_mode,
                                 label_as_signal=label_as_signal,
                                 accel_in_second_dim=accel_in_second_dim,
                                 cnn_1dinput_len=cnn_1dinput_len)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def sampleecg_get_datasets_for_train(data,
                                     load_train=True,
                                     load_test=True):
    """"
    Returns Sample ECG Dataset For Training Mode
    """

    eval_mode = False  # Test set includes validation normals
    label_as_signal = True

    accel_in_second_dim = True


    return sampleecg_get_datasets(data, 
                                  load_train, 
                                  load_test,
                                  eval_mode=eval_mode,
                                  label_as_signal=label_as_signal,
                                  accel_in_second_dim=accel_in_second_dim,
                                  cnn_1dinput_len=187)


def sampleecg_get_datasets_for_eval_with_anomaly_label(data,
                                                       load_train=True,
                                                       load_test=True):
    """"
    Returns Sample ECG Dataset For Evaluation Mode
    Label is anomaly status
    """

    eval_mode = True  # Test set includes validation normals
    label_as_signal = False

    accel_in_second_dim = True


    return sampleecg_get_datasets(data, 
                                  load_train, 
                                  load_test,
                                  eval_mode=eval_mode,
                                  label_as_signal=label_as_signal,
                                  accel_in_second_dim=accel_in_second_dim,
                                  cnn_1dinput_len=187)


def sampleecg_get_datasets_for_eval_with_signal(data,
                                                load_train=True,
                                                load_test=True):
    """"
    Returns Sample ECG Dataset For Evaluation Mode
    Label is signal
    """

    eval_mode = True  # Test set includes anormal samples as well as validation normals
    label_as_signal = True

    accel_in_second_dim = True

    return sampleecg_get_datasets(data, 
                                  load_train, 
                                  load_test,
                                  eval_mode=eval_mode,
                                  label_as_signal=label_as_signal,
                                  accel_in_second_dim=accel_in_second_dim,
                                  cnn_1dinput_len=187)


datasets = [
    {
        'name': 'SampleECG_ForTrain',
        'input': (187, 1),
        'output': ('signal'),
        'regression': True,
        'loader': sampleecg_get_datasets_for_train,
    },
    {
        'name': 'SampleECG_ForEvalWithAnomalyLabel',
        'input': (187, 1),
        'output': ('normal', 'anomaly'),
        'loader': sampleecg_get_datasets_for_eval_with_anomaly_label,
    },
    {
        'name': 'SampleECG_ForEvalWithSignal',
        'input': (187, 1),
        'output': ('signal'),
        'loader': sampleecg_get_datasets_for_eval_with_signal,
    }
]
