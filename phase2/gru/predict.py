#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : predict.py
@Author: XuYaoJian
@Date  : 2022/11/21 11:04
@Desc  : 
"""
import os

import numpy as np
import torch
from data_processed import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
torch.cuda.set_device(device)


def forecast_gru(settings):
    model_filenames = ['gru_o_60.pt']
    path = settings['filepath3']
    predictions = []

    for index, model_name in enumerate(model_filenames):
        name = path + model_name
        model = torch.load(name)  # 读取整个模型
        model.eval()
        train_features, _ = get_data(settings, settings['path'])
        seq_len = settings['input_len']
        data = np.array(train_features[-seq_len:]).astype('float32')
        data = torch.unsqueeze(torch.from_numpy(data), 0).to(device)
        with torch.no_grad():
            outputs = model(data)
        pred = outputs.detach().cpu().numpy()
        pred = np.clip(np.array(pred), a_min=0., a_max=1.).reshape(-1)
        pred = pred[-settings['step_size']:]*settings['scale'][2]
        pred[0] = 0.
        pred[14] = 0.
        predictions.extend(pred)
    print(predictions[:15])
    return predictions[:15]







