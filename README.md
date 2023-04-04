#### 基础环境：

GPU型号：Nvidia Tesla p100 Pcie 16g

CUDA Version 11.6

cudnn：8.1.0.77

cudatoolkit：11.3.1

python：3.9.12

```
torch==1.10.2
numpy==1.22.4
pandas==1.3.5
sklearn==0.0
scikit-learn==1.0.2
xgboost==1.4.2
```

#### 运行：

```shell
# 训练(大约20分钟)
sh train.sh
# 预测
sh predict.sh
```

榜上结果保存在文件夹final_result的final_result.csv中

复现结果保存在文件夹reproduce_result下的reproduce_result.csv中