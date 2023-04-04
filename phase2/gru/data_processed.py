#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : data_processed.py
@Author: XuYaoJian
@Date  : 2022/10/31 16:19
@Desc  : 
"""
import math

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.simplefilter('ignore')


def add_features(df):
    df = fill_data(df)
    df['hour_sin'] = df['Hour'].apply(lambda x: math.sin(x))
    df['hour_cos'] = df['Hour'].apply(lambda x: math.cos(x))
    df['Hour'] = (df['Hour'] - df['Hour'].min()) / (df['Hour'].max() - df['Hour'].min())  # 小时归一化
    df['Day'] = (df['Day'] - df['Day'].min()) / (df['Day'].max() - df['Day'].min())  # 天数归一化
    return df


def fill_data(df):
    # 1、Dir缺失值使用后值填充 161个缺失值
    # 2、Spd、Temp缺失值使用线性插补 161个缺失值，8个缺失值
    print(df.isnull().sum())
    df['Dir'].bfill(inplace=True)
    df['Dir'] = (df['Dir'] - df['Dir'].min()) / (df['Dir'].max() - df['Dir'].min())  # 风向归一化
    df['Spd'].interpolate(inplace=True)
    df['Temp'].interpolate(inplace=True)
    if df.isnull().any().any():
        print("数据有NAN值")
    else:
        return df


def get_data(settings, path):
    print("Loading train data")
    df = pd.read_csv(path)
    print('Adding features')
    df = add_features(df)
    cols = [c for c in df.columns if c not in settings['remove_features']]
    df = df[cols]
    print("data cols", df.columns)
    train_features = df.copy()
    train_targets = df['Radiation']
    return np.array(train_features), np.array(train_targets)
