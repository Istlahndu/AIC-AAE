### 项目结构

+ 团队编号： AIC-2025-22271428
+ Eval 文件夹为推理预测代码与预测结果
+ Large 文件夹为主文件夹
    + model.py: 模型定义文件
    + datapre.py: 数据处理文件
    + 其余为训练文件
+ logs 文件夹为程序日志输出
+ model2 文件夹存放模型权重文件
+ requirements 为模型训练环境

**请注意，模型文件夹对磁盘占用情况较大（6G）文档末有github链接，不含模型文件夹！**

### 训练命令

在 Large 文件夹下运行以下命令分别训练5000类和400类

```python
python pretrain1.py
python finetuning0.py
```

在 Eval 文件夹下运行以下命令预测5000类和400类

```python
python eval.py
```

预测结果存放在 Eval 文件夹下，根据提交要求已转移至以下文件夹：
+ Eval / AIC-2025-22271428-初赛预测结果
+ Eval / AIC-2025-22271428-初赛预测结果

### 相关说明
+ 本项目文件仅包含源代码与模型权重，docker镜像请在网盘链接中获取

+ 相关代码托管于github：https://github.com/Istlahndu/AIC-AAE

+ 代码库含有本项目的docker镜像：
https://github.com/Istlahndu/AIC-AAE/pkgs/container/aaetraining

+ 训练不使用脚本进行，执行训练代码请参照前文给出的命令，不再单独使用bash

+ 训练需要在多卡分布式进行，请确保有足够的容量存放batchsize

+ 由于docker镜像用于辅助复现，镜像内不含模型权重文件
