import torch
import torch.nn as nn
from torch import optim
import  matplotlib.pyplot as plt

# 等间隔学习率衰减
def dm01():
    # 定义变量，记录初始的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 创建数据集
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w，需要自动微分（求导）
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 创建优化器对象，动量法 ——> 加速模型的收敛，减少震荡
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 创建等间隔学习率衰减对象
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 创建两个列表，分别表示：训练轮数，每轮训练用的学习率
    lr_list, epoch_list = [], []

    # 循环遍历训练论述，进行具体训练
    for epoch in range(epochs):     # epoch: 0 ~ 199
        # 获取每轮训练的次数和学习率，并保存到列表中
        # .append(数据)是列表自带方法：把括号里的内容追加到列表的末尾。
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()) # 获取最后的lr(learning rate, 学习率)

        # 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            # 先计算预测值，然后基于损失桉树计算损失值
            y_pred = w * x
            # 计算损失，最小二乘法
            loss = (y_pred - y_true) ** 2
            # 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

    print(f'lr_list: {lr_list}')

    # 可视化（x轴：训练的轮数，y轴：每轮训练用的学习率）
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()


# 指定间隔学习率衰减
def dm02():
    # 定义变量，记录初始的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 创建数据集
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w，需要自动微分（求导）
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 创建优化器对象，动量法 ——> 加速模型的收敛，减少震荡
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 创建指定间隔学习率衰减对象
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)

    # 创建两个列表，分别表示：训练轮数，每轮训练用的学习率
    lr_list, epoch_list = [], []

    # 循环遍历训练论述，进行具体训练
    for epoch in range(epochs):  # epoch: 0 ~ 199
        # 获取每轮训练的次数和学习率，并保存到列表中
        # .append(数据)是列表自带方法：把括号里的内容追加到列表的末尾。
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  # 获取最后的lr(learning rate, 学习率)

        # 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            # 先计算预测值，然后基于损失桉树计算损失值
            y_pred = w * x
            # 计算损失，最小二乘法
            loss = (y_pred - y_true) ** 2
            # 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

    print(f'lr_list: {lr_list}')

    # 可视化（x轴：训练的轮数，y轴：每轮训练用的学习率）
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

# 指数学习率衰减
def dm03():
    # 定义变量，记录初始的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 创建数据集
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数w，需要自动微分（求导）
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 创建优化器对象，动量法 ——> 加速模型的收敛，减少震荡
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 创建等间隔学习率衰减对象
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 创建两个列表，分别表示：训练轮数，每轮训练用的学习率
    lr_list, epoch_list = [], []

    # 循环遍历训练论述，进行具体训练
    for epoch in range(epochs):  # epoch: 0 ~ 199
        # 获取每轮训练的次数和学习率，并保存到列表中
        # .append(数据)是列表自带方法：把括号里的内容追加到列表的末尾。
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  # 获取最后的lr(learning rate, 学习率)

        # 循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            # 先计算预测值，然后基于损失桉树计算损失值
            y_pred = w * x
            # 计算损失，最小二乘法
            loss = (y_pred - y_true) ** 2
            # 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

    print(f'lr_list: {lr_list}')

    # 可视化（x轴：训练的轮数，y轴：每轮训练用的学习率）
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()