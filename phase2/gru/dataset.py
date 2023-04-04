#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : dataset.py
@Author: XuYaoJian
@Date  : 2022/11/1 16:24
@Desc  : 
"""

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, features, targets, settings):
        super(TrainDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.input_len = settings['input_len']
        self.output_len = settings['output_len']

    def __len__(self):
        return len(self.features) - self.output_len - self.input_len + 1

    def __getitem__(self, index):
        # print(index)
        output_begin = index + self.input_len
        output_end = index + self.input_len + self.output_len
        return self.features[index: output_begin].astype('float32'), \
               self.targets[output_begin: output_end].reshape(-1).astype('float32')
