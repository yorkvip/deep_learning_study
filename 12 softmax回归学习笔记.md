# 1.12 Softmax 回归（从零实现）学习笔记

> 目标：用手写代码实现 softmax 回归，在 Fashion-MNIST 数据集上训练分类器。

---

## 一、整体流程概览

```
加载数据 → 初始化参数 → 定义模型(net) → 定义损失(cross_entropy) → 定义优化器(SGD)
→ 训练循环(train_epoch_ch3) → 评估精度(evaluate_accuracy)
```

---

## 二、核心函数逐个解析

### 1. `softmax(X)` — 把任意实数向量变成概率分布

```python
def softmax(X):
    X_exp = torch.exp(X)          # 对每个元素求 e^x，全部为正数
    partition = X_exp.sum(1, keepdim=True)  # 沿"行"方向求和，保持列形状
    return X_exp / partition       # 广播：每行除以该行的和
```

- **输入**：形状 `(批次大小, 类别数)` 的矩阵，元素是任意实数。
- **输出**：同样形状的矩阵，但**每行元素之和 = 1**，可理解为概率。
- **`keepdim=True`** 的作用：保持 `partition` 为 `(批次, 1)` 的列向量，这样才能和 `(批次, 类别)` 做广播除法。若不加，`sum(1)` 会变成一维 `(批次,)`，广播方向就错了。
- **广播机制**：`(n, 10) / (n, 1)` → 自动把分母复制 10 列，逐元素相除。

---

### 2. `net(X)` — 前向传播（线性层 + softmax）

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)
```

分步拆解：

| 步骤 | 代码 | 说明 |
|------|------|------|
| 展平图像 | `X.reshape((-1, w.shape[0]))` | `(批次,1,28,28)` → `(批次,784)`。`-1` 表示自动推断批次大小 |
| 线性变换 | `torch.matmul(..., w) + b` | 矩阵乘 `(批次,784)×(784,10)` = `(批次,10)`，再加偏置 `b(10,)`（广播到每行） |
| 转概率 | `softmax(...)` | 把输出变成概率分布 |

---

### 3. `cross_entropy(y_hat, y)` — 交叉熵损失

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
```

- **`y_hat[range(len(y_hat)), y]`**：高级索引，从每行取出"真实标签 y 对应位置"的预测概率。
  - 例：`y = [0, 2]`，`y_hat` 有 2 行 → 取第 0 行第 0 列、第 1 行第 2 列。
- **`-torch.log(...)`**：取负对数。预测概率越接近 1，损失越接近 0；越接近 0，损失趋近无穷大。
- **返回值**：形状 `(批次大小,)` 的**向量**（不是标量！），每个元素是一个样本的损失。

> 注意：因为返回的是向量，后续 `.backward()` 前必须先 `.sum()` 或 `.mean()` 聚合成标量。

---

### 4. `accuracy(y_hat, y)` — 计算单批预测正确数

```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 取每行最大值的索引 = 预测类别
    cmp = y_hat.type(y.dtype) == y    # 逐元素比较，得布尔张量
    return float(cmp.type(y.dtype).sum())  # True→1，求和 = 正确数量
```

- **`argmax(axis=1)`**：沿"列"方向找最大值的索引，即模型最看好的类别。
- **`type(y.dtype)`**：统一数据类型，避免比较出错。
- **返回**：正确预测的样本数（浮点数）。

---

### 5. `evaluate_accuracy(net, data_iter)` — 评估整个数据集的精度

```python
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()                        # 切换到评估模式（关闭 Dropout 等）
    metric = d2l.Accumulator(2)           # 2 个累加槽：[正确数, 总数]
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]          # 正确数 / 总数 = 准确率
```

- **`net.eval()`**：评估模式。训练用 `net.train()`，评估用 `net.eval()`，两者要配对。
- **`d2l.Accumulator(n)`**：一个 n 槽的累加器，`.add()` 依次加到各槽，`metric[i]` 取第 i 槽的值。

---

### 6. `train_epoch_ch3` — 训练一个 epoch（重点 + 有错需修正）

#### 你写的原始代码（含 bug）：

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()                                          # ❌ Bug 1
            updater.step()
            metric.add(float(1) * len(y), accuracy(y_hat, y), y.size.numel)  # ❌ Bug 2, 3
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.size.numel())    # ❌ Bug 4
    return metric[0] / metric[2], metric[1] / metric[2]
```

#### Bug 逐条分析：

| # | 位置 | 错误代码 | 正确写法 | 原因 |
|---|------|---------|---------|------|
| 1 | Optimizer 分支 | `l.backward()` | `l.mean().backward()` | `l` 是**向量**（每个样本一个损失值），PyTorch 只能对标量求梯度，必须先 `.mean()` 或 `.sum()` 聚合 |
| 2 | Optimizer 分支 | `float(1) * len(y)` | `float(l.sum())` | 第一个累加槽应存"**总损失**"，而不是"批次大小×1"。`float(1)*len(y)` 只是批次大小，完全丢掉了损失信息 |
| 3 | Optimizer 分支 | `y.size.numel` | `y.numel()` | `y.size` 返回 `torch.Size` 对象（本质是 tuple），**没有 `numel` 属性**。正确做法是直接调 `y.numel()`。而且原代码连括号 `()` 都漏了，即使属性存在也不会调用 |
| 4 | else 分支 | `y.size.numel()` | `y.numel()` | 同上，`torch.Size` 没有 `numel()` 方法 |

#### 修正后的正确代码：

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)  # 3 个槽：[总损失, 正确数, 总样本数]
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()       # 清空旧梯度
            l.mean().backward()       # ✅ 向量先取均值再反向传播
            updater.step()            # 更新参数
        else:
            l.sum().backward()        # 手写 updater：先求和再反向传播
            updater(X.shape[0])       # 按批次大小更新
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # ✅ 统一累加
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回 [平均损失, 平均准确率]
```

#### 两个分支的区别：

| | `torch.optim.Optimizer` 分支 | 手写 updater 分支（else） |
|---|---|---|
| 适用场景 | 用 PyTorch 内置优化器（如 `SGD`） | 自己实现的 `sgd` 函数 |
| 梯度聚合 | `l.mean()` — 按批次均值更新 | `l.sum()` — 按批次总和更新 |
| 参数更新 | `updater.step()` 一行搞定 | `updater(batch_size)` 手动更新 |
| 本质 | 两种方式数学上等价（学习率差一个批次倍数） | |

#### `metric` 三个槽的含义：

```
metric[0] = 所有批次的损失之和
metric[1] = 所有批次预测正确的样本数
metric[2] = 所有批次的样本总数
```

最终返回：
- `metric[0] / metric[2]` = **平均每个样本的损失**
- `metric[1] / metric[2]` = **准确率**

---

## 三、关键知识点速查

### `y.numel()` vs `y.size` vs `len(y)` vs `y.shape`

| 写法 | 返回类型 | 含义 | 示例（y 有 256 个元素） |
|------|---------|------|----------------------|
| `y.numel()` | `int` | 元素总个数 | `256` |
| `y.size` | `torch.Size` | 形状对象（类似 tuple） | `torch.Size([256])` |
| `len(y)` | `int` | 第 0 维的长度 | `256` |
| `y.shape` | `torch.Size` | 同 `y.size` | `torch.Size([256])` |

> 记忆口诀：要**个数**用 `numel()`，要**形状**用 `size`/`shape`，要**第一维长度**用 `len()`。

### `l.backward()` 为什么会报错？

PyTorch 的 `backward()` 默认只能对**标量**（1 个数）求梯度。当 `l` 是向量时：
```
RuntimeError: grad can be implicitly created only for scalar outputs
```
解决：`l.sum().backward()` 或 `l.mean().backward()`，先聚合成标量。

### `isinstance()` 的作用

```python
isinstance(updater, torch.optim.Optimizer)  # 判断 updater 是不是 PyTorch 优化器
isinstance(net, torch.nn.Module)            # 判断 net 是不是 nn.Module 模型
```
本代码用 `isinstance` 区分"用框架优化器"和"手写优化器"两条路径，实现兼容两种方式。

---

## 四、训练完整调用链

```python
# 1. 定义优化器（两种二选一）
lr = 0.1
trainer = torch.optim.SGD([w, b], lr=lr)   # 方式 A：框架优化器
# 或自己实现 sgd 函数 → 方式 B：手写 updater

# 2. 训练多个 epoch
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch_ch3(net, train_iter, cross_entropy, trainer)
    test_acc = evaluate_accuracy(net, test_iter)
    print(f'epoch {epoch}: loss={train_loss:.3f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}')
```

---

## 五、一句话总结每个函数

| 函数 | 一句话 |
|------|--------|
| `softmax(X)` | 把实数矩阵每行变成和为 1 的概率 |
| `net(X)` | 图像展平 → 线性变换 → softmax 概率 |
| `cross_entropy` | 取真实类别对应的预测概率，取负对数作为损失 |
| `accuracy` | 统计单批次预测正确的样本数 |
| `evaluate_accuracy` | 累加多批次，算整个数据集的准确率 |
| `train_epoch_ch3` | 遍历训练集一遍，前向→算损失→反向→更新参数，返回平均损失和准确率 |
