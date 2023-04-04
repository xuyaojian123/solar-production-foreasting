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
        'checkpoints': "../checkpoints/gru_reproduce/",
        'remove_features': ['Day', 'Dir','hour_cos'],
        'rnn_layer': 1,
        "input_len": 50,
        "output_len": 150,
        'batch_size': 32,
        'in_var': 5,
        'out_var': 1,
        "dropout": 0.05,
        'epoch_num': 80,
        'learning_rate': 0.01,
        "horizons": [60],
        "except_horizons":[150],
        "step_size": 30,
        'patience': 15,
        'random_seed': 2222,
    }

    return settings