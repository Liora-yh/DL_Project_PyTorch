import torch
import torch.nn as nn
from torch.nn import MaxPool2d


# 定义函数，演示单通道池化
def dm01():
    # 创建1个 1通道3*3的二维矩阵
    inputs = torch.tensor([[
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]])
    # [               # 1 通道C
    #   [             # 3 高度H
    #     [0, 1, 2],  # 3 宽度W
    #     [3, 4, 5],
    #     [6, 7, 8]
    #   ]
    # ]
    # print(f'inputs: {inputs}, shape: {inputs.shape}')   #(1, 3, 3)

    # 创建最大池化层
    # 参1：池化核（池化窗口）大小，参2：步长，参3：填充
    pool1 = nn.MaxPool2d(2, 1, 0)
    outputs1 = pool1(inputs)
    print(f'outputs1: {outputs1}, shape: {outputs1.shape}')

    # 创建平均池化层
    # 参1：池化核（池化窗口）大小，参2：步长，参3：填充
    pool2 = nn.AvgPool2d(2, 1, 0)
    outputs2 = pool2(inputs)
    print(f'outputs2: {outputs2}, shape: {outputs2.shape}')


# 定义函数，演示多通道池化
def dm02():
    # 创建1个 3通道3*3的二维矩阵
    inputs = torch.tensor([
        [               # 通道1，高度H、宽度W: 3、3
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ],
        [               # 通道2，高度H、宽度W: 3、3
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],
        [               # 通道2，高度H、宽度W: 3、3
            [11, 22, 33],
            [44, 55, 66],
            [77, 88, 99]
        ]
    ])
    # print(f'inputs: {inputs}, shape: {inputs.shape}')   #(3, 3, 3)

    # 创建最大池化层
    # 参1：池化核（池化窗口）大小，参2：步长，参3：填充
    pool1 = nn.MaxPool2d(2, 1, 0)
    outputs1 = pool1(inputs)
    print(f'outputs1: {outputs1}, shape: {outputs1.shape}')

    # 创建平均池化层
    # 参1：池化核（池化窗口）大小，参2：步长，参3：填充
    pool2 = nn.AvgPool2d(2, 1, 0)
    outputs2 = pool2(inputs)
    print(f'outputs2: {outputs2}, shape: {outputs2.shape}')


if __name__ == '__main__':
    dm01()
    dm02()