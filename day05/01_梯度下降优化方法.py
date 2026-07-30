import torch
import torch.nn as nn
from torch import optim


# 梯度下降优化方法——动量法（Momentum）
def dm01():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    print(criterion.shape)
    # 创建优化器（函数对象）——> 基于SGD（随机梯度下降），加入参数momentum，就是动量法
    # 参1：（待优化的）参数列表，参2：学习率，参3：动量参数
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)      # momentum=0(默认)，只考虑本次梯度
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    # 如果定义损失函数那里做求和了，那么反向传播那里就不用加sum()求和了
    # 即 criterion = ((w ** 2) / 2.0).sum() ——> criterion.backward()
    print(f'w: {w}, w.grad:{w.grad}')

    # 重复上述步骤，第2次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')


# 梯度下降优化方法——AdaGrad
def dm02():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    print(criterion.shape)
    # 创建优化器（函数对象）——> 基于Adagrad（自适应学习率下降）
    # 参1：（待优化的）参数列表，参2：学习率
    optimizer = optim.Adagrad(params=[w], lr=0.01)
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')

    # 重复上述步骤，第2次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')

# 梯度下降优化方法——RMSProp
def dm03():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    print(criterion.shape)
    # 创建优化器（函数对象）——> 基于RMSProp（自适应学习率下降）
    # 参1：（待优化的）参数列表，参2：学习率
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.9)
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')

    # 重复上述步骤，第2次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')

# 梯度下降优化方法——Adam
def dm04():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 定义损失函数
    criterion = ((w ** 2) / 2.0)
    print(criterion.shape)
    # 创建优化器（函数对象）——> 基于Adagrad（自适应学习率下降）
    # 参1：（待优化的）参数列表，参2：学习率
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))     #betas=(梯度用二点衰减系数, 学习率用的衰减系数)
    # 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')

    # 重复上述步骤，第2次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad:{w.grad}')

if __name__ == '__main__':
    dm01()
    print('_'*30)
    dm02()
    print('_' * 30)
    dm03()
    print('_' * 30)
    dm04()
