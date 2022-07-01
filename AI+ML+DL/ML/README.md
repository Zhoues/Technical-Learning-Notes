# ML(机器学习)

[TOC]



# 机器学习基础

## 数据

**数据的基本概念和数学表示**

- 数据整体叫做数据集（data set）
- 每一行数据称为一个样本（sample）
- 除最后一列表示种类，每一列表达样本的一个特征（feature）[属性、维度]
- 最后一列，称为标记（label）



- 最后一列之外的其他列构成样本特征矩阵`X`
- 最后一列构成标记向量`y`





第`i`个样本的特征向量写作
$$
X^{(i)}
$$
第`i`个样本第`j`个特征值写作
$$
X^{(i)}_{j}
$$
第`i`个样本的标记写作
$$
y^{(i)}
$$


- 数据构成的多维空间称为特征空间（feature space）

-  **分类任务本质就是在特征空间切分**
- 在高维空间同理



## 机器学习的基本任务

### 分类任务

- 二分类
- 多分类
  - 一些算法只支持完成二分类的任务
  - 但是多分类的任务可以转换成二分类的任务
  - 有一些算法天然可以完成多分类的任务
- 多标签分类

### 回归任务

- 结果是一个连续数字的值，而非一个类别
  - 有一些算法只能解决回归问题
  - 有一些算法只能解决分类问题
  - 有一些算法既可以解决回归问题，也可以解决分类问题

- 一些情况下，回归任务可以简化成分类任务



## 机器学习方法分类

### 监督学习

**给机器的训练数据拥有“标记”或者“答案”**

- k近邻
- 线性回归与多项式回归
- 逻辑回归
- SVM
- 决策树与随机森林

### 非监督学习

**对没有“标记”的数据进行分类——聚类分析**

- **对数据进行降维处理**：方便可视化
  - 特征提取
  - 特征压缩：PCA

- **异常检测**

### 半监督学习

一部分数据有“标记”或者“答案”，另一部分数据没有

通常都先使用无监督学习手段对数据做处理，之后使用监督学习手段做模型的训练和预测

### 增强学习

根据周围的情况，采取行动，更具采取行为的结果，学习行动方式



## 机器学习的其他分类

### 批量学习和在线学习

#### 批量学习

- 优点：简单

- 缺点：每次重新批量学习，运算量巨大

![](https://photo-bucket-1309504341.cos.ap-beijing.myqcloud.com/%E6%89%B9%E9%87%8F%E5%AD%A6%E4%B9%A0.png)

#### 在线学习

- 优点：及时反映新的环境的变化

- 问题：新的数据带来不好的变化？
  - 解决方案：需要加强对数据进行监控
- 其他：也适用于数据理巨大，完全无法批量学习的环境

![](https://photo-bucket-1309504341.cos.ap-beijing.myqcloud.com/%E5%9C%A8%E7%BA%BF%E5%AD%A6%E4%B9%A0.png)



### 参数学习和非参数学习

#### 参数学习

一旦学到了参数，就不再需要原有的数据集

#### 非参数学习

不对模型进行过多假设

**非参数学习不等于没有参数！**



# 工具简介

## Jupyter Notebook

### 魔法命令

**%run 命令**

```c
%run Python脚本相较于Jupyte rNotebook的位置
%run ML_Demo/main.py
```

把Python脚本加载进Notebook（包括函数等内容）



**当然也可以把该Python脚本用import进行引入**

```python
import ML_Demo.main
from ML_Demo import main
```



**%timeit 命令**（测试单条指令的运行时间，多次运算后选择最快的几次取平均值）

```python
%timeit L = [i**2 for i in range(1000000)]

> 210 ms ± 1.75 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```



**%%timeit 命令**（测试指令块的运行时间，多次运算后选择最快的几次取平均值）

```python
%%timeit
L = []
for i in range(1000):
    L.append(i ** 2)

> 226 µs ± 1.19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```



**%time**命令（测试单条指令的运行时间，只运行一次）

```python
%time L = [i**2 for i in range(1000000)]

> CPU times: total: 219 ms
> Wall time: 223 ms
```



**%%time**命令（测试单条指令的运行时间，只运行一次）

```python
%%timeit
L = []
for i in range(1000):
    L.append(i ** 2)

> CPU times: total: 0 ns
> Wall time: 9.01 ms
```



## Numpy

### 创建Numpy数组

```python
import numpy as np

# 构造多维数组(shape中的元组表示维度，dtype表示类型[默认为浮点型]，fill_value为填充数组的值)
np.zeros(10)
> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

np.zeros(10, dtype=int)
> [0 0 0 0 0 0 0 0 0 0]

np.zeros(shape=(3, 4), dtype=int)
> [[0 0 0 0]
> [0 0 0 0]
> [0 0 0 0]]

np.ones(shape=(3, 4))
>[[1. 1. 1. 1.]
> [1. 1. 1. 1.]
> [1. 1. 1. 1.]]

np.full(shape=(3, 4), fill_value=666)
> [[666 666 666 666]
> [666 666 666 666]
> [666 666 666 666]]



# 快速构造递增数组(和python3的range基本一致，但是步长可以为小数)
np.arange(0, 10)
> [0 1 2 3 4 5 6 7 8 9]

np.arange(0, 10, 0.5)
> [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5 9.  9.5]

np.arange(10, 0, -1)
> [10  9  8  7  6  5  4  3  2  1]

np.arange(10, 0, -1.5)
> [10.   8.5  7.   5.5  4.   2.5  1. ]



# 快速等分区间(和python3的range的参数基本一致，但是第三个参数是分成区间的个数)
np.linspace(0, 10, 5)
> [ 0.   2.5  5.   7.5 10. ]




# 快速生成随机整型向量(low下限, high上限, size生成的个数, dtype类型)
np.random.randint(low=0, high=10, size=1, dtype=int)
> [4]

np.random.randint(low=0, high=10, size=(2, 3), dtype=int)
>[[9 0 3]
> [2 5 0]]

# 可以修改随机数的种子(随机数的生成和种子是一一对应的关系)
np.random.seed = xxx

# 快速生成随机浮点型向量
np.random.random(10)
> [0.81430166 0.60066306 0.86614656 0.76750252 0.07353262 0.68018871 0.03070963 0.9244369  0.1550115  0.69099267]

np.random.random((3, 3))
> [[0.95475793 0.89230311 0.73095396]
> [0.47903551 0.34646541 0.49917552]
> [0.58268326 0.63736238 0.77798554]]

# 随机生成一个标准正态分布的值
np.random.normal()
> 0.10850831101316975

# 随机生成一个自定义正态分布的值(第一个loc为均值，第二个scale为方差,最后一个size为生成数组的样式)
np.random.normal(10,100)
> 112.9574899167099

np.random.normal(0, 1, size=(2, 3))
> [[-0.51506516 -0.31596675  0.61156275]
> [-0.66518676  0.46993595  0.8913645 ]]
```



### Numpy.array的基本操作

```python
import numpy as np

x = np.arange(10)
xx = np.arange(15).reshape(3, 5)
print(x)
print(xx)

> [0 1 2 3 4 5 6 7 8 9]
>[[ 0  1  2  3  4]
> [ 5  6  7  8  9]
> [10 11 12 13 14]]

# 查询维度		ndim
x.ndim
> 1
xx.ndim
> 2

# 查询形状		shape
x.shape
> (10,)
xx.shape
> (4,5)

# 查询元素个数	size
x.size
> 10
xx.size
> 15
```



### Numpy.array的数据访问

```python
# 下标直接访问(特别注意多位数组的访问方式)		x[_x]	xx[ _x, _y]
x[1]
> 1
x[-1]
> 9

xx[2][2] = xx[(2, 2)] = xx[2,2]
> 12

# 切片(三个参数和range基本一致,特别注意多位数组的切片)—————————可以降维，无法升维
x[2:5]
> [2 3 4]
x[:5]
> [0 1 2 3 4]
x[5:]
> [5 6 7 8 9]
x[::2]
> [[0 2 4 6 8]]
x[::-1]
> [9 8 7 6 5 4 3 2 1 0]

xx[:2, :3]
> [[0 1 2]
> [5 6 7]]

xx[::-1, ::-1]
> [[14 13 12 11 10]
> [ 9  8  7  6  5]
> [ 4  3  2  1  0]]

# 深拷贝(如果简单的采用切片，得到的是浅拷贝)
xxx = xx[:2, :3].copy()

# 数组重构形状		reshape——————————————无法降维，可以升维
x.reshape(2, 5)
> [[0 1 2 3 4]
> [5 6 7 8 9]]

x.reshape(5, -1)	# 只关心行数
> [[0 1]
>  [2 3]
>  [4 5]
>  [6 7]
>  [8 9]]

x.reshape(-1, 5)	# 只关心列数
> [[0 1 2 3 4]
>  [5 6 7 8 9]]
```



### Numpy.array的合并与分割

```python
import numpy as np

x = np.arange(3)
xx = x[::-1].copy()
print(x)
print(xx)

> [0 1 2 ]
> [2 1 0]

# 合并
## 使用concatenate([])来进行合并矩阵，但是注意，必须是同一个ndim维度的
np.concatenate([x, xx])
> [0 1 2 2 1 0]

y = np.np.arange(6).reshape(2, 3)
> [[0 1 2]
> [3 4 5]]

np.concatenate([y, y])
> [[0 1 2]
> [3 4 5]
> [0 1 2]
> [3 4 5]]

## 用axis控制合并维度
np.concatenate([y, y], axis=1)
> [[0 1 2 0 1 2]
> [3 4 5 3 4 5]]

np.concatenate([x.reshape(1, -1), y]
> [[0 1 2]
> [0 1 2]
> [3 4 5]]      
               
## 使用 vstack([]) 来进行列向的堆叠，不一定需要是同一个ndim维度的，只需要列数相同即可
## 使用 hstack([]) 来进行横向的堆叠，不一定需要是同一个ndim维度的，只需要行数相同即可
np.vstack([x, y])
> [[0 1 2]
> [0 1 2]
> [3 4 5]]
               
## 使用tile完成向量的堆叠成矩阵(vstack,hstack的合并使用)    
np.title(x,[y,z])	# 把x向量进行横向堆叠y次，纵向堆叠z次
           
               
# 分割               
## 使用split(x, [y1, y2,...])来进行分割，对x进行切割，已y1,y2...为切割点
x = np.arange(10)
np.split(x, [2, 7])      
> [array([0, 1]), array([2, 3, 4, 5, 6]), array([7, 8, 9])]            
## 同样用axis控制分割维度
## 使用vsplit(x, []) 来进行列向的分割
## 使用hsplit(x, []) 来进行横向的分割
```



### Numpy.array中运算

```python
# 支持矩阵与数之间的算数运算(对应元素相乘)

## python3中对list进行乘法操作实际上是进行复制多次
## numpy可以直接对array进行乘法操作，而且比列表生成式要快的多的多
x = np.arange(10)
2 * x
> [ 0  2  4  6  8 10 12 14 16 18]

## 可以直接对numpy.array进行所有python3支持的算术运算，结果表现为对数组的每一个元素进行该运算

## numpy.array支持的特殊的运算
## np.abs(x)			x所有元素取绝对值
## np.sin(x)			x所有元素取正弦值
## np.cos(x)			x所有元素取余弦值
## np.tan(x)			x所有元素取正切值
## np.exp(x)			x所有元素取e的幂
## np.power(n,x)		x所有元素的n次幂
## np.log()			x所有元素取e的对数
## np.log2()			x所有元素取2的对数
## np.log10()		x所有元素取10的对数




# 支持矩阵之间的算数运算(对应元素相乘)
x = np.arange(1, 11).reshape((2, 5))
y = np.random.randint(2, 10, (2, 5))
x + y
x * y


# 支持向量与矩阵的运算(等价于该向量对于矩阵的每一个行或者列进行了算数运算)
v = np.array([1, 2])
a = np.arange(4).reshape((2, 2))
> [1 2]
>[[0 1]
> [2 3]]

v + a
> [[1 3]
> [3 5]]

v * q
> [[0 2]
> [2 6]]


# 矩阵的标准乘法
A.dot(B)

# 矩阵的转置
A.T

# 矩阵的逆
np.linalg.inv(A)

# 矩阵的伪逆矩阵
np.linalg.pinv(A)
```



### Numpy中的聚合操作

```python
# 对数组进行求和
np.sum(A)
# 对数组进行求积
np.prod(A)
# 最大值最小值
np.max(A)   np.min(A)
# 平均值
np.mean(A)
# 中位数
np.median(A)
# 查看百分位的值(表示v中比x%大的最小值)
np.percentile(v, x)
# 方差
np.var(A)
# 标准差
np.std(A)

# 对于二维矩阵，如果单独进行行列统计，需要添加axio参数
# axis=0：表示沿着行对每个列进行操作
# axis=1：表示沿着列对每个行进行操作


```



### Numpy中排序及其索引

```python
# 找到数组最小值的索引
np.argmin()
# 找到数组最大值的索引
np.argmax()

# 排序数组
np.sort(A,axis=0)	# 对列进行排序(沿着行)
np.sort(A,axis=1)	# 对行进行排序(沿着列)

# 排序索引数组(返回的是排序之后元素在原先数组中的位置)
np.argsort(A)

# 利用快速排序的分割算法划分位置(左半边比x小，右半边比x大)
np.partition(A,x)	
# 利用快速排序的分割算法划分位置(左半边比x小，右半边比x大)(返回的是划分之后元素在原先数组中的位置)
np.argpartition(A,x)	
```





