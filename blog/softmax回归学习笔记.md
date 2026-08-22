# Softmax 回归学习笔记

## 1. 什么是 Softmax 回归



## 2. Softmax 公式

$$\hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$

## 3. 交叉熵损失

$$L = -\log(\hat{y}_{\text{真实类别}})$$

## 4. 从0实现 vs 简洁实现

| | 从0实现 | 简洁实现 |
|---|---|---|
| 模型 | `softmax(matmul(X, w) + b)` | `nn.Sequential(nn.Flatten(), nn.Linear(784, 10))` |
| 损失函数 | 手写 `cross_entropy` | `nn.CrossEntropyLoss()` |
| 优化器 | 手写 `sgd([w, b], lr, batch_size)` | `torch.optim.SGD(net.parameters(), lr)` |
| 参数 | 手动定义 `w`、`b` | `net.parameters()` 自动获取 |

## 5. 训练结果



## 6. 踩过的坑



## 7. 核心代码速查

```python
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
