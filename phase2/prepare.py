#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : prepare.py
@Author: XuYaoJian
@Date  : 2022/11/17 22:04
@Desc  : 
"""


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        'checkpoints': "../checkpoints/gru/",
        "filepath1": "./checkpoints/xgb2/",
        "filepath2": "./checkpoints/xgb_final1/",
        "filepath3": "./checkpoints/gru/",
        'remove_features': ['Day', 'Dir', 'hour_cos'],
        'rnn_layer': 1,
        "input_len": 50,
        "output_len": 150,
        'batch_size': 32,
        'in_var': 5,
        'out_var': 1,
        "dropout": 0.05,
        'epoch_num': 80,
        'learning_rate': 0.01,
        "horizons": [30, 60, 90, 120],
        "except_horizons": [150],
        "step_size": 30,
        'patience': 15,
        'scale': [1.08, 1.8, 2, 0.7, 1.3, 1.3, 0.85, 0.6, 1, 0.5],
        'path': "./data/origin.csv",
        'random_seed': 2222,
    }

    return settings
