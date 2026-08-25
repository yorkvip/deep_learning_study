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


def _init_plot(num_epochs):
    """创建三合一实时绘图，返回 (fig, axes, lines)"""
    import matplotlib.pyplot as plt
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('config', 'InlineBackend.figure_format = "png"')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = ['train loss', 'train acc', 'test acc']
    lines = []
    for ax, lbl in zip(axes, labels):
        line, = ax.plot([], [], label=lbl)
        lines.append(line)
        ax.set_xlabel('epoch')
        ax.set_ylabel(lbl)
        ax.legend(loc='best')
        ax.set_xlim(1, num_epochs)
    fig.tight_layout()
    return fig, axes, lines


def _update_plot(fig, axes, lines, data, epoch, num_epochs):
    """更新绘图数据并刷新"""
    from IPython import display
    for i, (line, d) in enumerate(zip(lines, data)):
        line.set_data(range(1, len(d) + 1), d)
        if d:
            ymin, ymax = min(d), max(d)
            pad = (ymax - ymin) * 0.1 or 0.1
            axes[i].set_ylim(ymin - pad, ymax + pad)
        axes[i].set_title(f'{line.get_label()}  (epoch {epoch+1}/{num_epochs})')
    display.clear_output(wait=True)
    display.display(fig)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """CPU 训练，调用方式不变"""
    if isinstance(net, torch.nn.Module):
        net.train()
    fig, axes, lines = _init_plot(num_epochs)
    data = [[], [], []]

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        data[0].append(train_loss)
        data[1].append(train_acc)
        data[2].append(test_acc)
        _update_plot(fig, axes, lines, data, epoch, num_epochs)

    display.clear_output(wait=True)
    display.display(fig)
    print(f'done — loss {train_loss:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}')
    return train_loss, train_acc


def train_ch3_gpu(net, train_iter, test_iter, loss, num_epochs, updater):
    """GPU 训练：自动把数据一次性预加载到 GPU，避免每个 batch 重复传输"""
    from IPython import display
    device = torch.device('cuda')
    net = net.to(device)

    # 预加载全部数据到 GPU
    X_train, y_train = [], []
    for X, y in train_iter:
        X_train.append(X.to(device))
        y_train.append(y.to(device))
    X_train, y_train = torch.cat(X_train), torch.cat(y_train)

    X_test, y_test = [], []
    for X, y in test_iter:
        X_test.append(X.to(device))
        y_test.append(y.to(device))
    X_test, y_test = torch.cat(X_test), torch.cat(y_test)
    batch_size = next(iter(train_iter))[0].shape[0]

    if isinstance(net, torch.nn.Module):
        net.train()
    fig, axes, lines = _init_plot(num_epochs)
    data = [[], [], []]
    n = len(X_train)

    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        total_loss, correct, count = 0.0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            y_hat = net(X_train[idx])
            l = loss(y_hat, y_train[idx])
            updater.zero_grad()
            l.backward()
            updater.step()
            total_loss += l.item() * y_train[idx].numel()
            correct += (y_hat.argmax(1) == y_train[idx]).sum().item()
            count += y_train[idx].numel()

        train_loss, train_acc = total_loss / count, correct / count
        net.eval()
        with torch.no_grad():
            test_acc = (net(X_test).argmax(1) == y_test).float().mean().item()
        net.train()

        data[0].append(train_loss)
        data[1].append(train_acc)
        data[2].append(test_acc)
        _update_plot(fig, axes, lines, data, epoch, num_epochs)

    display.clear_output(wait=True)
    display.display(fig)
    print(f'done — loss {train_loss:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}')
    return train_loss, train_acc
