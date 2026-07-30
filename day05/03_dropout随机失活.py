import torch
import torch.nn as nn


def dm01():
    # 创建隐藏层输出结果
    t1 = torch.randint(0, 10, size=(1, 4)).float()
    print(f't1: {t1}')

    # 进行下一层加权求和 和 激活函数计算
    linear1 = nn.Linear(4, 5)
    # 加权求和
    l1 = linear1(t1)
    print(f'l1: {l1}')
    # 激活函数
    output = torch.relu(l1)
    print(f'output: {output}')

    # 对激活函数进行随机失活dropout处理 ——> 只有训练阶段有，测试阶段没有
    dropout = nn.Dropout(p=0.4)     # 每个神经元都有40%的概率被kill
    # 具体的随机失活动作
    d1 = dropout(output)
    print(f'dl(随机失活后的数据): {d1}')    # 未被失活的进行缩放，缩放比例为：1/(1-p)

if __name__ == '__main__':
    dm01()