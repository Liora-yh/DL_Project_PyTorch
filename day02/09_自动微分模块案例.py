"""
    循环实现计算梯度，更新参数

    需求：
        求 y = x ** 2 + 20 的极小值，并打印y是最小值时，w的值（梯度）
"""

import torch

w = torch.tensor(10, requires_grad=True, dtype=torch.float)

loss = w ** 2 + 20     # 求导：2w

# 利用梯度下降法，循环迭代1000求最优解
print(f'开始权重初始值w: {w}, (0.01 * w.grad): 无, loss: {loss}')  # 10, 无, 120
# 迭代100次，求最优解
for i in range(1, 101):
    # 正向计算（前向传播）
    loss = w ** 2 + 20
    # 梯度清零 w.grad.zero_()
    # 至此（第一次的时候），还没有计算梯度，所以w.grad=None，要做非空判断
    if w.grad is not None:
        w.grad.zero_()
    # 反向传播
    loss.sum().backward()
    # 梯度更新
    w.data = w.data - 0.01 * w.grad
    # 打印本次梯度更新后的权重参数结果
    print(f'第{i}次，权重初始值w: {w}, (0.01 * w.grad): {0.01 * w.grad:.5f}, loss: {loss}')
    # :.5f：格式化约束，把计算结果保留5 位小数再输出
    # print(f'梯度值为：{w.grad}')

# 打印最终结果
print(f'最终结果 权重：{w}, 梯度：{w.grad}, loss：{loss}')



