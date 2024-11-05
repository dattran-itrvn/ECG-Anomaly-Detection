###################################################################################################
#
# Copyright (C) 2024 ITRVN. All Rights Reserved.
# This software is proprietary to ITRVN and its licensors.
#
###################################################################################################
"""
Main classes and functions for The PTB Diagnostic ECG Database
"""
import math
import os
import pickle

import numpy as np
import torch
from numpy.fft import fft
from torch.utils.data import Dataset

import pandas as pd
import scipy

from utils.dataloader_utils import makedir_exist_ok


class ECG_DataFrame_Parser(Dataset):  # pylint: disable=too-many-instance-attributes
    """
    The base dataset class for ECG data used in Heartbeat Categorization.
    Includes main preprocessing functions.
    Expects a dataframe with common_dataframe_columns.
    """

    common_dataframe_columns = ["file_identifier", "raw_data", "label"]


    def process_file_and_return_signal_windows(self, file_raw_data):


        # First dimension: 1
        # Second dimension: number of windows
        # Third dimension: Window for self.duration_in_sec. 1000 samples for default settings
        file_cnn_signals = np.expand_dims(file_raw_data, axis=0)
        file_cnn_signals = file_cnn_signals.transpose([1, 0, 2])

        return file_cnn_signals

    def create_common_empty_df(self):
        """
        Create empty dataframe
        """
        df = pd.DataFrame(columns=self.common_dataframe_columns)
        return df

    def __init__(self, root, d_type,
                 transform=None,
                
                 eval_mode=False,
                 label_as_signal=True,
                 train_ratio=0.8,
                 cnn_1dinput_len=188,
                 main_df=None
                 ):

        if d_type not in ('test', 'train'):
            raise ValueError(
                "d_type can only be set to 'test' or 'train'"
                )

        self.main_df = main_df
        self.df_normals = self.main_df[main_df['label'] == 0]
        self.df_anormals = self.main_df[main_df['label'] == 1]
        self.train_ratio = train_ratio

        self.root = root
        self.d_type = d_type
        self.transform = transform

        self.eval_mode = eval_mode
        self.label_as_signal = label_as_signal

        self.num_of_features = 3

        self.cnn_1dinput_len = cnn_1dinput_len

        processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        self.processed_folder = processed_folder

        makedir_exist_ok(self.processed_folder)

        self.specs_identifier = f'eval_mode_{self.eval_mode}_' + \
                                f'label_as_signal_{self.label_as_signal}_'

        train_dataset_pkl_file_path = \
            os.path.join(self.processed_folder, f'train_{self.specs_identifier}.pkl')

        test_dataset_pkl_file_path =  \
            os.path.join(self.processed_folder, f'test_{self.specs_identifier}.pkl')

        if self.d_type == 'train':
            self.dataset_pkl_file_path = train_dataset_pkl_file_path

        elif self.d_type == 'test':
            self.dataset_pkl_file_path = test_dataset_pkl_file_path

        self.signal_list = []
        self.lbl_list = []

        self.__create_pkl_files()
        self.is_truncated = False

    def __create_pkl_files(self):
        if os.path.exists(self.dataset_pkl_file_path):

            print('\nPickle files are already generated ...\n')

            (self.signal_list, self.lbl_list) = pickle.load(open(self.dataset_pkl_file_path, 'rb'))
            return

        self.__gen_datasets()

    def normalize_signal(self, features):
        """
        Normalize signal with Local Min Max Normalization
        """
        # Normalize data:
        for instance in range(features.shape[0]):
            instance_max = np.max(features[instance, :, :], axis=1)
            instance_min = np.min(features[instance, :, :], axis=1)

            for feature in range(features.shape[1]):
                for signal in range(features.shape[2]):
                    features[instance, feature, signal] = (
                        (features[instance, feature, signal] - instance_min[feature]) /
                        (instance_max[feature] - instance_min[feature])
                    )

        return features

    def __gen_datasets(self):

        train_features = []
        test_normal_features = []

        for _, row in self.df_normals.iterrows():
            raw_data = row['raw_data']
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)

            num_training = int(self.train_ratio * cnn_signals.shape[0])

            for i in range(cnn_signals.shape[0]):
                if i < num_training:
                    train_features.append(cnn_signals[i])
                else:
                    test_normal_features.append(cnn_signals[i])

            
        train_features = np.asarray(train_features)
        test_normal_features = np.asarray(test_normal_features)

        anomaly_features = []

        for _, row in self.df_anormals.iterrows():
            raw_data = row['raw_data']
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)

            for i in range(cnn_signals.shape[0]):
                anomaly_features.append(cnn_signals[i])


        anomaly_features = np.asarray(anomaly_features)

        train_features = self.normalize_signal(train_features)
        test_normal_features = self.normalize_signal(test_normal_features)
        anomaly_features = self.normalize_signal(anomaly_features)

        # ARRANGE TEST-TRAIN SPLIT AND LABELS
        if self.d_type == 'train':
            self.lbl_list = [train_features[i, :, :] for i in range(train_features.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)

            if not self.label_as_signal:
                self.lbl_list = np.zeros([len(self.signal_list), 1])

        elif self.d_type == 'test':
            # Testing in training phase includes only normal test samples
            if not self.eval_mode:
                test_data = test_normal_features
            else:
                test_data = np.concatenate((test_normal_features, anomaly_features), axis=0)

            self.lbl_list = [test_data[i, :, :] for i in range(test_data.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)
           
            if not self.label_as_signal:
                self.lbl_list = np.concatenate(
                                    (np.zeros([len(test_normal_features), 1]),
                                     np.ones([len(anomaly_features), 1])), axis=0)
        # Save pickle file
        pickle.dump((self.signal_list, self.lbl_list), open(self.dataset_pkl_file_path, 'wb'))

    def __len__(self):
        if self.is_truncated:
            return 1
        return len(self.signal_list)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        if self.is_truncated:
            index = 0

        signal = self.signal_list[index]
        lbl = self.lbl_list[index]

        if self.transform is not None:
            signal = self.transform(signal)

            if self.label_as_signal:
                lbl = self.transform(lbl)

        if not self.label_as_signal:
            lbl = lbl.astype(np.int32)
        else:
            lbl = lbl.numpy().astype(np.float32)

        return signal, lbl
