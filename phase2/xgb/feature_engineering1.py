#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : feature_engineering1.py
@Author: XuYaoJian
@Date  : 2022/10/30 15:06
@Desc  : 
"""
import math
import pandas as pd
import numpy as np


def fill_data(df):
    # 1、Dir缺失值使用后值填充 161个缺失值
    # 2、Spd、Temp缺失值使用线性插补 161个缺失值，8个缺失值
    print(df.isnull().sum())
    df['Dir'].bfill(inplace=True)
    df['Dir'] = (df['Dir'] - df['Dir'].min()) / (df['Dir'].max() - df['Dir'].min()) #归一化
    df['Spd'].interpolate(inplace=True)
    df['Temp'].interpolate(inplace=True)
    if df.isnull().any().any():
        print("数据有NAN值")
    else:
        return df

def get_train_data(path):
    df = pd.read_csv(path)
    df = fill_data(df)
    print('Adding features')
    df = add_features(df)
    # df.to_feather('features.f')
    print("Loading train data finish")
    return df


def add_features(df):
    # df = make_rolling_features(df)
    # df = make_rolling_features_2(df)
    df = make_rolling_features_3(df)
    # df = df.interpolate().fillna(method='bfill')
    # df['hour_sin'] = df['Hour'].apply(lambda x: math.sin(x))
    # df['hour_cos'] = df['Hour'].apply(lambda x: math.cos(x))
    # df['Hour'] = df['Hour'].astype('category')
    return df

def make_rolling_features_2(df):
    for fea in ['Spd', 'Radiation']:
        for i in [2, 3, 5, 7, 9, 12, 15]:
            df[fea + "_rolling_max" + str(i)] = df[fea].transform(lambda x: x.rolling(i).max())
            df[fea + "_rolling_std" + str(i)] = df[fea].transform(lambda x: x.rolling(i).std())
            df[fea + "_rolling_mean" + str(i)] = df[fea].transform(lambda x: x.rolling(i).mean())

    for fea in ['Hour','Temp','Dir', 'Spd','Radiation']:
        for i in range(1, 16):
            df[fea + "_past" + str(i)] = df[fea].shift(i)  # 往后移

    df = df[~df['Radiation_past' + str(15)].isnull()].reset_index(drop=True)
    return df

def make_rolling_features_3(df): #最好的
    for fea in ['Temp', 'Radiation']:
        for i in [2, 3, 5, 7, 9, 12, 15]:
            df[fea + "_rolling_max" + str(i)] = df[fea].transform(lambda x: x.rolling(i).max())
            df[fea + "_rolling_std" + str(i)] = df[fea].transform(lambda x: x.rolling(i).std())
            df[fea + "_rolling_mean" + str(i)] = df[fea].transform(lambda x: x.rolling(i).mean())

    for fea in ['Hour', 'Radiation']:
        for i in range(1, 16):
            df[fea + "_past" + str(i)] = df[fea].shift(i)  # 往后移

    df = df[~df['Radiation_past' + str(15)].isnull()].reset_index(drop=True)
    return df


def make_rolling_features(df):
    # for fea in ['Dir', 'Spd', 'Temp', 'Radiation']:
    for fea in ['Dir', 'Spd', 'Temp', 'Radiation']:
        # df[fea + "_simple_diff"] = df[fea].diff(1)
        # df[fea + "_simple_diff1Day"] = df[fea].diff(150)
        # for i in [6, 12, 36, 72, 144]:
        # for i in [3, 6, 9, 12, 15]:
            # df[fea + "_rolling_mean" + str(i)] = df[fea].transform(lambda x: x.rolling(i).mean())
            # df[fea + "_rolling_max" + str(i)] = df[fea].transform(lambda x: x.rolling(i).max())
            # df[fea + "_rolling_min" + str(i)] = df[fea].transform(lambda x: x.rolling(i).min())
            # df[fea + "_rolling_std" + str(i)] = df[fea].transform(lambda x: x.rolling(i).std())
            # df[fea + "_rolling_mean_diff" + str(i)] = df[fea + "_rolling_mean" + str(i)].diff(1)
            # df[fea + "_rolling_mean_diffn" + str(i)] = df[fea + "_rolling_mean" + str(i)].diff(150)
            # df[fea + "_rolling_mean_cal" + str(i)] = df[fea + "_rolling_mean" + str(i)] - df[fea]
        for i in [1, 3, 6, 9, 12, 15]:
            if i != 1:
                df[fea + "_rolling_max" + str(i)] = df[fea].transform(lambda x: x.rolling(i).max())
                df[fea + "_rolling_std" + str(i)] = df[fea].transform(lambda x: x.rolling(i).std())
                df[fea + "_rolling_mean" + str(i)] = df[fea].transform(lambda x: x.rolling(i).mean())
        for i in range(1, 6):
            df[fea + "_diff" + str(i)] = df[fea].diff(i)
            df[fea + "_past" + str(i)] = df[fea].shift(i)  # 往后移

    df = df[~df['Temp_rolling_mean' + str(15)].isnull()].reset_index(drop=True)
    return df


def add_target(df):
    print("adding targeting")
    index = 150
    for i in range(index):
        df['target' + str(i+1)] = df["Radiation"].shift(-(i+1)) #向上移动

    # # 尝试的一步，298条数据，150个模型
    # df = df.drop(df[(df['Hour'] != 20)].index)
   # 划分出训练集和验证集和测试集
   #  need_predict = df[-1:]
    df = df[~df['target' + str(index)].isnull()].reset_index(drop=True)
    print("adding targeting finish")

    # cols = [c for c in need_predict.columns if ('target' not in c)]
    # need_predict = need_predict[cols]

    return df#, need_predict


def split_data(df):
    print("splitting data")

    df_train = df[(df['Day'] <= 269)].reset_index(drop=True)
    df_val = df[(df['Day'] > 269) & (df['Day'] <= 279)].reset_index(drop=True)
    df_test = df[(df['Day'] > 279) & (df['Day'] <= 289)].reset_index(drop=True)

    cols = [c for c in df.columns if ('target' not in c)]
    x_train, x_val, x_test = df_train[cols], df_val[cols], df_test[cols]

    cols = [c for c in df.columns if 'target' in c]
    print(cols)

    y_train, y_val, y_test = df_train[cols], df_val[cols], df_test[cols]
    print("splitting data finish")
    return x_train, x_val, x_test,  y_train, y_val, y_test


def get_predictX(df):
    index = 150
    for i in range(index):
        df['target' + str(i + 1)] = df["Radiation"].shift(-(i + 1))  # 向上移动
    need_predict = df[-1:]
    cols = [c for c in df.columns if ('target' not in c)]
    need_predict = need_predict[cols]
    need_predict = np.array(need_predict)
    return need_predict

