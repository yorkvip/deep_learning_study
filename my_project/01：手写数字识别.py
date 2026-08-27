# 这是我做的第一个项目，梦开始的地方~

#导入必要的库
import os, sys
sys.path.insert(0, os.path.abspath('..'))  # 把父目录加入路径，才能导入 good_functions
import torch
from torch import nn
from good_functions import *

#加载数据
from torchvision import transforms
import torchvision

mnist_train = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor(), download=True)

#一些数据的定义
batch_size,num_inputs,num_hiddens,num_outputs=256,28*28,256,10

#把这些转换成可以训练的数据集
from torch.utils.data import DataLoader

train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter  = DataLoader(mnist_test,  batch_size=batch_size, shuffle=False)

#初始化隐藏层的参数
w1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
w2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)

b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params=[w1,b1,w2,b2]

#激活函数RELU
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)

def net(X):
    X = X.reshape(X.shape[0], -1)   # (batch,1,28,28) → (batch,784)
    result1=relu(X@w1+b1)
    return (result1 @ w2+b2)
#损失函数：reduction='none' 返回逐样本损失，供从0实现求sum/mean
loss=nn.CrossEntropyLoss(reduction='none')

#softmax函数
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition  #应用了广播机制

def accuracy(y_hat, y):
    "计算预测正确的数量"
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 取每行最大值的索引作为预测类别
    cmp = y_hat.type(y.dtype) == y    # 预测类别与真实标签比较
    return float(cmp.type(y.dtype).sum())  # 返回正确预测的数量
def evaluate_accuracy(net,data_iter):
    "计算指定数据集上的模型精度"
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

#接下来开始写训练函数
num_epochs,lr=5,0.1

def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric =d2l.Accumulator(2)
    for X, y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0]/metric[1]
def updater(batch_size):
    return d2l.sgd(params,lr,batch_size)

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
    train_acc=train_metrics
    test_acc=evaluate_accuracy(net,test_iter)
    return test_acc,train_acc

# ============ GPU 预加载训练版（用你的手写模型和参数）============
# 思路：整个数据集一次性搬到显存，训练时用 randperm 在 GPU 上取 batch，全程零传输。
def train_gpu(num_epochs, lr, seed=42):
    global w1, b1, w2, b2, params
    torch.manual_seed(seed)                                # 固定随机种子，结果可复现
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    # 一次性把整个数据集搬到 GPU
    X_train = torch.stack([d[0] for d in mnist_train]).to(device)
    y_train = torch.tensor([d[1] for d in mnist_train]).to(device)
    X_test  = torch.stack([d[0] for d in mnist_test]).to(device)
    y_test  = torch.tensor([d[1] for d in mnist_test]).to(device)

    # 把手写参数搬到 GPU（注意从当前计算的图上 detach，搬过去只作为新参数）
    w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, device=device)*0.01, requires_grad=True)
    w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, device=device)*0.01, requires_grad=True)
    b1 = nn.Parameter(torch.zeros(num_hiddens, device=device), requires_grad=True)
    b2 = nn.Parameter(torch.zeros(num_outputs, device=device), requires_grad=True)
    params = [w1, b1, w2, b2]

    num_train = X_train.shape[0]
    for epoch in range(num_epochs):
        perm = torch.randperm(num_train, device=device)   # GPU 上打乱
        train_loss = train_correct = 0.0
        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            y_hat = net(X_train[idx])
            l = loss(y_hat, y_train[idx])
            l.sum().backward()                            # 自动求导
            with torch.no_grad():                         # 手写 SGD 更新 + 清梯度
                for p in params:
                    p.data -= lr * p.grad / batch_size
                    p.grad.zero_()
            train_loss += float(l.sum().detach())   # detach 脱离计算图，再转标量不告警
            train_correct += accuracy(y_hat, y_train[idx])

        test_correct = 0.0
        for i in range(0, X_test.shape[0], batch_size):
            test_correct += accuracy(net(X_test[i:i+batch_size]), y_test[i:i+batch_size])

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'epoch {epoch+1:3d} | train loss {train_loss/num_train:.4f} | '
                  f'train acc {train_correct/num_train:.4f} | test acc {test_correct/X_test.shape[0]:.4f}')
    print(f'\n完成 {num_epochs}轮: 最终测试准确率 {test_correct/X_test.shape[0]:.4f}')
    # 保存参数，小游戏直接加载，避免每次重新训练
    torch.save({'w1': w1.detach().cpu(), 'b1': b1.detach().cpu(),
                'w2': w2.detach().cpu(), 'b2': b2.detach().cpu()}, 'mnist_params.pt')
    print('参数已保存到 mnist_params.pt')

# 跑 100 轮，想调参就改这里两个参数
# 加 __main__ 保护：直接运行本文件才训练；被小游戏导入时只取函数，不重复训练
if __name__ == '__main__':
    train_gpu(100, 0.01)






