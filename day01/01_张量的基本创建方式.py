"""
张量的借本创建方式
张量 ——> 存储同一类型元素的容器，且元素值必须是数值才可以
"""
import numpy as np
import torch
from numpy.ma.core import size
from torch.nn.utils import prune


# 1、torch.tensor 根据指定数据创建张量
def dm01():
    # 场景1：标量 张量
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('_' * 30)
    # f'' 叫做 f-string 格式化字符串,作用：直接在字符串里嵌入变量 / 表达式，不用复杂拼接。

    # 场景2：二维列表 ——> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('_' * 30)

    # 场景3： numpy nd数组 ——> 张量
    # NumPy 函数，生成指定区间内的随机整数，区间规则：左闭右开 [low, high)
    # 标准语法：np.random.randint(low, high=None, size=None, dtype=int)
    # low：下界（包含），如果不写 high，区间变成 [0, low)
    # high（可选）：上界（不包含），写上后区间为 [low, high)
    # size（可选）：控制输出形状：数字 => 一维数组；元组 => 多维数组；不写 => 单个数字
    # dtype（可选）：指定整数类型，默认 int
    data = np.random.randint(0, 10, size=(2, 3))
    # t3 = torch.tensor(data)
    t3 = torch.tensor(data, dtype=torch.float)
    print(f't3: {t3}, type: {type(t3)}')



# 2、torch.Tensor 根据形状创建张量
def dm02():
    # 场景1：标量 张量
    t1 = torch.Tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('_' * 30)

    # 场景2：二维列表 ——> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('_' * 30)

    # 场景3： numpy nd数组 ——> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    # t3 = torch.tensor(data)
    t3 = torch.Tensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('_' * 30)

    # 场景4：尝试直接创建 指定维度（例如：2行3列的）张量
    t4 = torch.Tensor(2, 3)
    print(f't4: {t4}, type: {type(t4)}')


# 3、torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 根据指定类型创建张量
def dm03():
    # 场景1：标量 张量
    t1 = torch.IntTensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('_' * 30)


    # 场景2：二维列表 ——> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('_' * 30)

    # 场景3： numpy nd数组 ——> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('_' * 30)

    # 场景4：如果类型不匹配，会尝试自动转换类型
    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)       # 默认：float32【32位，占4字节】
    print(f't4: {t4}, type: {type(t4)}')

# 定义测试函数
if __name__ == '__main__':
    dm01()
    # dm02()
    # dm03()
    # pass






















