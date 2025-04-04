# 基于三层神经网络的CIFAR-10图像分类项目

## 项目简介
本项目旨在通过手工搭建三层神经网络分类器，实现对CIFAR-10数据集的图像分类任务。该项目不使用PyTorch、TensorFlow等现成的支持自动微分的深度学习框架，仅使用NumPy进行计算。

## 数据集
本项目使用CIFAR-10数据集，该数据集包含60000张32x32的彩色图像，分为10个不同的类别，每个类别有6000张图像。其中50000张图像用于训练，10000张图像用于测试。数据集请解压后放在该项目目录`/cifar-10-batches-py`下。


## 项目结构
```
.
├── model.py             # 模型架构
├── main.py              # 参数查找和训练模型
├── test.py              # 测试已有参数在测试集上的准确率
├── visualize.py         # 可视化已有参数
├── cifar-10-batches-py  # 数据集文件
    ├── data_batch_1
    ├── ...
├── README.md            # 项目说明文件
└── ckpts                # 可视化结果和模型参数
```

## 环境要求
- Python 3.8及以上版本
- NumPy 1.21.2及以上版本
- Matplotlib 3.4.3及以上版本
- pickle模块（Python内置）


## 参数查找和训练模型
1. 确保你已经下载了CIFAR-10数据集，并将其解压到名为`cifar-10-batches-py`的文件夹中，且该文件夹与代码文件在同一目录下。
2. 打开终端或命令提示符，进入项目所在目录。
3. 运行以下命令开始训练模型（如要的话可以修改文件中超参数的的选择范围和迭代次数）：
```bash
python main.py
```
4. 训练过程中，你会看到每100次迭代的训练损失值。训练结束后，模型的最优参数将保存到`ckpts/best_model.pkl`文件中，同时训练过程中的可视化图像（训练集和验证集的loss曲线、验证集的accuracy曲线）将保存到`ckpts`文件夹中。

## 测试模型
1. 训练完成后，代码会自动加载保存的最优模型参数，并在测试集上进行测试。
2. 如果需要根据已有模型参数进行测试，可以运行test.py，并指定其中的model_path。最优模型在测试集上获得的分类准确率为50.33%。
```bash
python test.py
```


## 模型权重下载
训练好的模型权重已上传至[百度云盘链接](https://pan.baidu.com/s/1F5qhgBmwNKCqmXXgkv-o_w?pwd=iabe)。可以下载该权重文件，并将其放置在ckpts目录下，然后运行测试代码`test.py`即可得到测试结果。
