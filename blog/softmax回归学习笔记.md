# Softmax 回归学习笔记

## 1. 什么是 Softmax 回归

Softmax 回归是用于**多分类问题**的线性模型。与线性回归输出连续值不同，它输出的是各类别的概率分布（所有类别概率加起来等于 1）。

- 输入：一张 28×28 的灰度图片，展平为 784 维向量
- 输出：10 个类别的概率（如 T恤、裤子、套衫...）
- 预测：取概率最大的类别作为结果

## 2. Softmax 公式

$$\hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$

- $o_j$ 是模型第 j 个类别的原始输出（logit）
- 用 `exp` 的原因：保证所有输出为正数，且放大差异
- 分母求和：保证概率加起来等于 1

## 3. 交叉熵损失

$$L = -\log(\hat{y}_{\text{真实类别}})$$

- 真实类别概率越高，loss 越小（接近 0）
- 真实类别概率越低，loss 越大（趋近无穷）
- **为什么不用 MSE**：MSE 在 softmax 输出接近 0 或 1 时梯度会饱和，训练极慢；交叉熵的梯度是 $\hat{y} - y$，不会饱和

## 4. 从0实现 vs 简洁实现

| | 从0实现 | 简洁实现 |
|---|---|---|
| 模型 | `softmax(matmul(X, w) + b)` | `nn.Sequential(nn.Flatten(), nn.Linear(784, 10))` |
| 损失函数 | 手写 `cross_entropy` | `nn.CrossEntropyLoss()` |
| 优化器 | 手写 `sgd([w, b], lr, batch_size)` | `torch.optim.SGD(net.parameters(), lr)` |
| 参数 | 手动定义 `w`、`b` | `net.parameters()` 自动获取 |

**关键区别**：`nn.CrossEntropyLoss()` 内部自带 softmax，所以模型的最后一层只需要输出 logits（原始分数），不需要手动做 softmax。

## 5. 训练结果

- 10 个 epoch，测试准确率约 **85%**
- 这是线性模型的天花板（Softmax 回归本质是线性分类器）
- 要进一步提升准确率，需要引入非线性 → 多层感知机（MLP）或 CNN

## 6. 踩过的坑

1. **accuracy 函数参数顺序写反**：定义 `accuracy(y, y_hat)` 但调用 `accuracy(net(X), y)`，导致维度不匹配报错。d2l 原版签名是 `accuracy(y_hat, y)`，调用时顺序要对应。

2. **d2l.Animator 在 PyCharm Jupyter 里卡住**：Animator 是实时绘图动画，PyCharm Jupyter 不支持。解决方案：把 `animator.add()` 替换成 `print()`，每个 epoch 打印一次结果。

3. **DataLoader num_workers 多进程问题**：Windows 下 `num_workers > 0` 在脚本模式需要 `if __name__ == '__main__'` 保护。在 Jupyter notebook 里不受影响，但如果把代码抽成 .py 脚本运行会报错。

## 7. 核心代码速查

```python
# 简洁实现版本
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
```
