import torch

# 只有标量张量才能求导，且大多数底层操作的都是 浮点型，记得转型
# 定义变量，记录：初始的权重w(旧)
# 参数1：初始值，参数2：是否自动微分（求导），参数3：数据类型
w = torch.tensor(10, requires_grad=True, dtype=torch.float)

# 定义1oss变量，表示损失函数
loss = 2 * w ** 2   # loss = 2w² ——> 求导：4w

# 打印梯度函数类型（了解）
# print(f'梯度函数类型：{type(loss.grad_fn)}')
# print(loss.sum())

# 计算梯度，梯度= 损失函数的导数，计算完后回记录到w.grad属性中
# loss.sum().backward()   # 保证loss是一个标量
loss.backward()         # 这里因为y本身就是标量，可以不写sum()

# 代入权重更新公式： W新 = W旧 - 学习率 * 梯度
w.data = w.data - 0.01 * w.grad     # 10 - 0.01 * (4 * 10) = 9.6

# 打印最终结果
print(f'更新后的权重：{w}')