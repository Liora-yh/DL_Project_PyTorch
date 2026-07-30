import torch
import torch.nn as nn

# 处理二维数据
def dm01():
    # 1张图片，2个通道，每个通道3行4列（像素点）
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print(f'input_2d: {input_2d}')

    # 创建批量归一化层（BN层）
    # 参1：输入特证数 = 图片的通道数
    # 参2：噪声数（小常数），默认为1e-5
    # 参3：动量值，用于计算移动平均统计量的动量值
    # 参4：表示使用可学习的变化参数（λ, β）对归一化（标准化）后的数据进行缩放和平移
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)

    # 对数据进行批量化处理
    output_2d = bn2d(input_2d)
    print(f'output_2d: {output_2d}')

# 处理一维数据
def dm02():
    # 创建样本数据
    # 2行2列，2条样本，每个样本有2个特征
    input_1d = torch.randn(size=(2, 2))

    # 创建线性层
    linear1 = nn.Linear(2, 4)

    # 对数据进行线性变换
    l1 = linear1(input_1d)
    print(f'l1: {l1}')

    # 创建批量归一化层（BN层）
    bn1d = nn.BatchNorm1d(num_features=4)

    # 对数据进行批量化处理
    output_1d = bn1d(l1)
    print(f'output_1d: {output_1d}')


if __name__ == '__main__':
    dm01()
    dm02()