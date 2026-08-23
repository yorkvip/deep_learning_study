# Softmax 回归学习笔记

## 1. 什么是 Softmax 回归

首先，softmax 回归是一种**分类**。在本轮代码中，通过训练，让模型能将图和图上的标签相匹配。

具体来说，模型给每个标签（在本次数据集有 10 个标签）打上分，然后给每个分数转化成 e^x ——这样可以对分数进行一个突出的效果。然后除以每一个元素的 e^x 之和。我们的预测标签 = 最高的那个分数对应的标签，这样就实现了分类。

## 2. Softmax 公式

$$\hat{y}_j = \frac{\exp(o_j)}{\sum_{k=1}^K \exp(o_k)}$$

- `o_j`：模型对第 j 个类别的原始评分（logit）
- `K`：类别总数（本次为 10）
- 分母：保证所有输出之和为 1，即概率分布

把每个评分（可能有正有负）用 e^x 放入同一个体系，使所有输出为正数且和为 1。

## 3. 线性运算与权重形状

图片是 28×28 的灰度图，展开成一个 784 维的向量。模型的线性运算为：

$$\mathbf{o} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

其中：
- **X** 形状：`(n, 784)`，n 为 batch size
- **W** 形状：`(784, 10)`，将 784 维输入映射到 10 个类别
- **b** 形状：`(10,)`
- **o** 形状：`(n, 10)`，每行是一张图片的 10 个类别评分

## 4. 交叉熵损失

$$L = -\sum_{i=1}^K y_i \log(\hat{y}_i)$$

- `y_i`：真实标签的 one-hot 编码（正确类别为 1，其余为 0）
- `y_hat_i`：模型预测的第 i 个类别概率

因为 y_i 只有正确类别那一项为 1，其余为 0，所以实际计算时简化为：

$$L = -\log(\hat{y}_{\text{正确类别}})$$

该损失函数的梯度非常简洁：

$$\frac{\partial L}{\partial \mathbf{o}} = \hat{\mathbf{y}} - \mathbf{y}$$

预测越准确，y_hat 越接近 y，梯度越小。这个非常形象——预测错了梯度大（大幅修正），预测对了梯度小（微调即可）。

## 5. 具体实现步骤

### 第一步：加载数据集

运用 `d2l.load_data_fashion_mnist` 函数，用 `train_iter`、`test_iter` 来接收，就是训练集和测试集。其中，`train_iter` 是训练集迭代器，元素为 `(图像批次, 标签批次)`。测试集元素是一样格式的元组。

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

### 第二步：把图片展开成向量

由于我们的线性矩阵运算不支持图片，接下来需要把图片展开成一个向量。图片是 28×28 的，那么就是一个 m 为 784 的矩阵。这个在后面的 `net` 函数里会处理：

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)
```

经过 `wX + b` 之后的值为标签数 = 10，所以权重矩阵 `w` 的形状 = `(784, 10)`。

### 第三步：初始化 w 和 b

在这个课程里，我们都是用均值为 0、方差为 0.01 对这两个矩阵中的参数值初始化。方差取 0.01 而不是 1，是为了让初始输出的评分接近 0，避免 softmax 后概率分布过于极端（某一类接近 1，其余接近 0），导致初始 loss 过大、梯度不稳定。

```python
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 第四步：定义 softmax 函数

把每个评分（可能有正有负）用 e^x 放入同一个体系：

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 应用了广播机制
```

### 第五步：定义损失函数

在这里的损失函数是交叉熵损失函数。根据梯度函数，该损失函数的梯度就是预测标签 `y_hat - y_true`，这个非常形象。显然，损失越小说明 softmax 函数分类得越对。

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
```

### 第六步：定义评估函数

我们用 `accuracy(y_hat, y)` 函数和 `evaluate_accuracy(net, data_iter)` 函数，来评估每次拟合后的正确率。

- `accuracy` 函数：对单个 batch 的 `y_hat` 计算正确的数量
- `evaluate_accuracy` 函数：在整个训练集或测试集里计算正确率

```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

### 第七步：开始训练

我们要写一个训练函数。首先，创建一个累加器 `Accumulator(3)`，三个槽分别对应累加**总损失、正确数、样本数**。

每个 batch 的训练流程：
1. 调用 `net(X)` 完成前向传播
2. 计算损失 `l = loss(y_hat, y)`
3. 调用 `l.mean().backward()`，对损失取平均后自动求导（取 mean 的目的是把 `l` 聚合成一个标量）
4. 根据求导结果，对参数进行更新

照这样训练多轮，每一轮后返回训练集正确率和测试集正确率。

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)          # 前向传播
        l = loss(y_hat, y)      # 计算损失
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()  # 反向传播
            updater.step()       # 更新参数
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
```

参数更新使用随机梯度下降（SGD）：

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \frac{\partial L}{\partial \mathbf{W}}$$

其中 η 为学习率（本次设为 0.1）。

## 6. 训练结果

训练 10 个 epoch 后，测试准确率约 **85%**。这是线性模型的天花板，要进一步提升需要引入非线性（多层感知机或 CNN）。

## 7. 从0实现 vs 简洁实现

| | 从0实现 | 简洁实现 |
|---|---|---|
| 模型 | `softmax(matmul(X, w) + b)` | `nn.Sequential(nn.Flatten(), nn.Linear(784, 10))` |
| 损失函数 | 手写 `cross_entropy` | `nn.CrossEntropyLoss()` |
| 优化器 | 手写 `sgd([w, b], lr, batch_size)` | `torch.optim.SGD(net.parameters(), lr)` |
| 参数 | 手动定义 `w`、`b` | `net.parameters()` 自动获取 |

**关键区别**：`nn.CrossEntropyLoss()` 内部自带 softmax，所以模型的最后一层只需输出 logits（原始评分），不需要手动做 softmax。

## 8. 踩过的坑

1. **accuracy 函数参数顺序写反**：定义 `accuracy(y, y_hat)` 但调用 `accuracy(net(X), y)`，导致维度不匹配报错。正确签名应是 `accuracy(y_hat, y)`。

2. **d2l.Animator 在 PyCharm Jupyter 里卡住**：Animator 是实时绘图动画，PyCharm Jupyter 不支持。解决方案：把 `animator.add()` 替换成 `print()`。

3. **l.mean() vs l.sum()**：从0实现用 `l.sum().backward()` 配合手写 sgd（内部会除以 batch_size）；简洁实现用 `l.mean().backward()` 配合框架优化器。两者数学等价，不能混用。
