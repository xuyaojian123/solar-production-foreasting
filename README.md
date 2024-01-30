#### 问题背景：

![enter image description here](https://img.alicdn.com/imgextra/i1/O1CN01veq6B820JjVW8C5Hv_!!6000000006829-2-tps-316-149.png)
图1 用户用电来源示意图

如图1所示，用户日常用电由两部分组成：太阳能发电和从电力市场购买。太阳能发电量受环境太阳能量辐射强度有关，表示为P(W)=![enter image description here](https://img.alicdn.com/imgextra/i3/O1CN012Hd3Ft1ghjx1iGj1C_!!6000000004174-2-tps-14-19.png)EA，其中E为太阳光辐射强度，单位为W/![enter image description here](https://img.alicdn.com/imgextra/i3/O1CN01gapQGm1cDjXXJ4ETU_!!6000000003567-2-tps-16-21.png),A=2为太阳能电池板面积，单位为![enter image description here](https://img.alicdn.com/imgextra/i3/O1CN01gapQGm1cDjXXJ4ETU_!!6000000003567-2-tps-16-21.png),![enter image description here](https://img.alicdn.com/imgextra/i3/O1CN012Hd3Ft1ghjx1iGj1C_!!6000000004174-2-tps-14-19.png)=0.5为太阳能电池板转换效率。当太阳能供给不足时，用户从电力市场购买额外电能。

问题：给定用户平均每日总用电需求，以及与太阳能生产量相关的历史环境信息，对用户未来太阳能生产量进行预测https://tianchi.aliyun.com/competition/entrance/532022/information

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
