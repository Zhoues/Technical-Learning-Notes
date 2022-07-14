# Pytorch

[TOC]

![机器学习构成元素](./picture/机器学习构成元素.png)



# Pytorch入门基础

## Tensor的创建

![Tensor的创建](./picture/Tensor的创建.png)



## Tensor的属性

![Tensor的属性](./picture/Tensor的属性.png)

```python
import torch

dev = torch.device('cpu')
a = torch.tensor([2, 2], dtype=torch.float, device=dev)
print(a)

> tensor([2., 2.])
```



## Tensor的稀疏张量

![Tensor的稀疏张量](./picture/Tensor的稀疏张量.png)

```python
import torch
# 确定稀疏张量中值的坐标，第一个列表为一维，第二个列表是二维，以此类推
i = torch.tensor([[0, 1, 2], [0, 1, 2]])
# 确定稀疏张量中值的大小
v = torch.tensor([1, 2, 3], dtype=torch.float)
# 生成稀疏张量，最后一个参数为稀疏张量的shape
a = torch.sparse_coo_tensor(i, v, (4, 4))
# 稀疏张量转化为紧凑张量
b = a.to_dense()
# 紧凑张量转化为稀疏张量
c = b.to_sparse_coo()

```



## Tensor的排序操作

![Tensor的排序操作](./picture/Tensor的排序操作.png)

![Tensor的有界性](./picture/Tensor的有界性.png)

## Tensor的统计学相关操作

![Tensor的创建](./picture/Tensor统计学相关函数.png)



![Tensor统计学相关函数2](./picture/Tensor统计学相关函数2.png)



## Tensor中的分布函数

![Tensor中的分布函数](./picture/Tensor中的分布函数.png)

## Tensor中的随机抽样

![Tensor中的随机抽样](./picture/Tensor中的随机抽样.png)

## Tensor中的范数运算

![Tensor中的范数运算](./picture/Tensor中的范数运算.png)

![Tensor的创建2](./picture/Tensor的创建2.png)

## Tensor中的矩阵分解

![Tensor中的矩阵分解](./picture/Tensor中的矩阵分解.png)

### PCA

![Tensor中的矩阵分解2](./picture/Tensor中的矩阵分解2.png)

### LDA

![Tensor中的矩阵分解3](./picture/Tensor中的矩阵分解3.png)



## Tensor的张量剪裁

![Tensor的张量剪裁](./picture/Tensor的张量剪裁.png)

## Tensor的索引和数据筛选

![Tensor的索引和数据筛选](./picture/Tensor的索引和数据筛选.png)

## Tensor的组合/拼接

![Tensor的组合与拼接](./picture/Tensor的组合与拼接.png)

## Tensor的切片

![Tensor的切片](./picture/Tensor的切片.png)

## Tensor的变形

![Tensor的变形](./picture/Tensor的变形.png)

## Tensor的填充与傅里叶变换

### 填充

![Tensor的填充与傅里叶变换](./picture/Tensor的填充与傅里叶变换.png)

### 频谱

![Tensor的填充与傅里叶变换2](./picture/Tensor的填充与傅里叶变换2.png)

## Pytorch简单编程技巧

### 模型的保存与加载

![模型的保存与加载](./picture/模型的保存与加载.png)

### 并行化

![并行化](./picture/并行化.png)

### Tensor与numpy的相互转化

![Tensor与numpy之间的相互转化](./picture/Tensor与numpy之间的相互转化.png)

## Pytorch与autograd

![叶子张量](./picture/叶子张量.png)

![grad](./picture/grad.png)

![backward](./picture/backward.png)

![grad2](./picture/grad2.png)

![Function](./picture/Function.png)





## torch.nn库

![torch.nn库](./picture/torch.nn库.png)

![nn.Parameter](./picture/nn.Parameter.png)

![torch.nn库2](./picture/torch.nn库2.png)

![torch.nn库3](./picture/torch.nn库3.png)

![torch.nn库4](./picture/torch.nn库4.png)

![torch.nn库5](./picture/torch.nn库5.png)

![torch.nn库6](./picture/torch.nn库6.png)

![torch.nn库7](./picture/torch.nn库7.png)

![torch.nn库8](./picture/torch.nn库8.png)



# 神经网络

## 基本概念

![神经网络的基本概念](./picture/神经网络的基本概念.png)

![神经网络的基本概念2](./picture/神经网络的基本概念2.png)

![感知器](./picture/感知器.png)



### 前向运算

![前向运算](./picture/前向运算.png)



### 反向传播

![反向传播](./picture/反向传播.png)





## 利用神经网络解决分类和回归问题

![利用神经网络解决分类和回归问题](./picture/利用神经网络解决分类和回归问题.png)















