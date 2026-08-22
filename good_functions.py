"""
good_functions.py
只包含 d2l 库里没有、但教材中需要手写的函数。

用法:
    from good_functions import *
"""

import torch
from IPython import display
from d2l import torch as d2l


def data_iter(batch_size, features, labels):
    """手动批量数据迭代器（d2l 没有）

    参数:
        batch_size (int):        每批样本数
        features (torch.Tensor):  特征矩阵
        labels (torch.Tensor):    标签向量

    生成:
        (X_batch, y_batch): 每次产出一批数据
    """
    import random
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def softmax(X):
    """对矩阵 X 每行做 softmax（d2l 没有）

    参数:
        X (torch.Tensor): 形状 (batch_size, num_classes)，任意实数

    返回:
        torch.Tensor: 同形状，每行和为 1（概率分布）
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def cross_entropy(y_hat, y):
    """交叉熵损失（d2l 没有，d2l 的 nn.CrossEntropyLoss 内部自带 softmax）

    参数:
        y_hat (torch.Tensor): 预测概率，形状 (batch_size, num_classes)
        y (torch.Tensor):      真实标签，形状 (batch_size,)，值为类别索引

    返回:
        torch.Tensor: 每个样本的损失值（向量）
    """
    return -torch.log(y_hat[range(len(y_hat)), y])


def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的平均准确率（d2l 没有）

    参数:
        net:       模型（函数或 nn.Module）
        data_iter: 数据迭代器

    返回:
        float: 平均准确率（0~1）
    """
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练一个 epoch（d2l 没有）

    参数:
        net:        模型（函数或 nn.Module）
        train_iter: 训练集数据迭代器
        loss:       损失函数
        updater:     参数更新器（函数 或 torch.optim.Optimizer）

    返回:
        (float, float): (平均损失, 平均准确率)
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        # 兼容两种损失函数：
        #   nn.CrossEntropyLoss() 返回标量均值 → 需乘 batch_size 还原为总损失
        #   手写 cross_entropy 返回向量 → l.sum() 就是总损失
        batch_loss = l.sum().item() * y.numel() if l.dim() == 0 else l.sum().item()
        metric.add(batch_loss, d2l.accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型多个 epoch（d2l 没有）

    参数:
        net:          模型
        train_iter:   训练集迭代器
        test_iter:    测试集迭代器
        loss:         损失函数
        num_epochs (int): 训练轮数
        updater:       参数更新器

    返回:
        (train_loss, train_acc)
    """
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    return train_loss, train_acc
