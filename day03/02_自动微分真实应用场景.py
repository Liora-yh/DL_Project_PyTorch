"""
1、先前向传播（正向传播）计算出预测值(z)
2、基于损失函数，结合预测值(z) 和 真实值(y) 来计算梯度
3、结合权重更新公式 w新 = w旧 - 学习率 * 梯度，来更新权重
"""
import torch

################### 定义x，表示：特征（输入数据），假设：2行5列，全1矩阵
x = torch.ones(2, 5)
print(f'x: {x}, type: {type(x)}')

################### 定义y，表示：标签（真实值），假设：2行3列，全0矩阵
y = torch.zeros(2, 3)
print(f'y: {y}, type: {type(y)}')

# 因为y是2行3列所以有2个样本，每一次会有3个预测结果，
# 然后由于x有5列有个样本有5个特征，所以w就需要有5列和特征的数量对应相乘，
# 然后w需要3行每行和特征对应相乘总共得到3个结果
# 有2*5的特征,需要得到2*3的结果,那么权重参数为5*3
################ 初始化（可自动微分的）权重 和 偏置
w = torch.randn(5, 3, requires_grad=True)   # x @ w + b =====> 要求x的列=w的行
# (2, 5) * (5, 3) ——> (2, 3)
print(f'w: {w}')

b = torch.randn(3, requires_grad=True)
print(f'b: {b}')

########## 前向传播（正向传播），计算出预测值(z)
z = torch.matmul(x, w) + b  # z = x @ w + b
print(f'z: {z}')

########## 定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z, y)

############ 进行自动微分，求导，结合反向传播，更新权重
loss.backward()

######## 打印w, b用来更新的梯度
print(f'w的梯度: {w.grad}')
print(f'b的梯度: {b.grad}')

