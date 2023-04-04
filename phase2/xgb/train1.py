#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : train1.py
@Author: XuYaoJian
@Date  : 2022/11/1 10:32
@Desc  :
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from feature_engineering1 import *
from sklearn.metrics import mean_squared_error
import operator

warnings.simplefilter('ignore')

if __name__ == "__main__":
    # name = "xgb2"
    name = "xgb2_reproduce"
    df = get_train_data("../data/origin.csv")
    df = add_target(df)
    pred_list = []
    test_list_mse = []
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df)
    # 训练1-150
    for i in range(0, 150):
        fixed_param = {
            'booster': 'gbtree',
            "gamma": 2,
            "min_child_weight": 10,
            'objective': 'reg:squarederror',
            'max_depth': 5,  # [3, 5, 6, 7, 9, 12, 15, 17, 25]#5
            'learning_rate': 0.01,  # 0.05
            "colsample_bytree": 0.33,#0.33
            'subsample': 0.33,  # 0.33
            "eval_metric": 'mae',  # mse,
            "gpu_id": 2,
            "tree_method": 'gpu_hist',
            'seed': 88,
        }
        model_name = "model_0_" + str(i + 1)
        label_name = 'target' + str(i + 1)
        print(f"------------------train  {model_name}---------------------------")

        feature_cols = x_train.columns

        dtrain = xgb.DMatrix(x_train, label=y_train[label_name], feature_names=feature_cols)
        dval = xgb.DMatrix(x_val, label=y_val[label_name], feature_names=feature_cols)
        dtest = xgb.DMatrix(x_test, label=y_test[label_name], feature_names=feature_cols)

        # dpredcit = xgb.DMatrix(need_predict, feature_names=feature_cols)

        model = xgb.train(fixed_param, dtrain, num_boost_round=1000,
                          evals=[(dval, 'eval'), (dtrain, 'train')],
                          early_stopping_rounds=30, verbose_eval=10)

        importance = model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
        print(model_name + ' feature_importance...')
        print(importance[:15])

        prediction_test_Y = model.predict(dtest)
        # 评估模型
        test_true = y_test[label_name]
        mse = mean_squared_error(prediction_test_Y, test_true)

        print("测试集mse为:"+str(mse))

        test_list_mse.append(mse)

        model.save_model("../checkpoints/"+name+"/" + model_name)

        importance = pd.DataFrame({
            'Radiation': importance[:15]
        })
        importance.to_csv('../checkpoints/'+name+'/importance_'+model_name+".csv", index=False)

    test_list_mse = np.array(test_list_mse)
    mean = test_list_mse.mean()
    print("均值为：")
    print(test_list_mse.mean())

    importance = pd.DataFrame({
        'test_mse': test_list_mse
    })
    importance.to_csv('../checkpoints/'+name+'/mse1-150_' + str(mean) + ".csv", index=False)

    fixed_param = pd.DataFrame({
        'fixed_param': fixed_param
    })
    fixed_param.to_csv('../checkpoints/'+name+'/fixed_param.csv', index=False)

