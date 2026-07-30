import torch
import numpy as np
from numpy.testing.print_coercion_tables import print_new_cast_table


# 场景1： 张量 ——> numpy nd数组对象
def dm01():
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f't1: {t1}, type: {type(t1)}')

    # 共享内存
    # n1 = t1.numpy()
    # 不共享内存
    n2 = t1.numpy().copy()
    # print(f'n1: {n1}, type: {type(n1)}')
    print(f'n2: {n2}, type: {type(n2)}')

    # n1[0] = 100
    # print(f'n1: {n1}')      # 【100, 2, 3, 4, 5]
    # print(f't1: {t1}')      # 【100, 2, 3, 4, 5]

    n2[0] = 100
    print(f'n2: {n2}')  # 【100, 2, 3, 4, 5]
    print(f't1: {t1}')  # 【1, 2, 3, 4, 5]

# 场景2： numpy nd 数组 ——> 张量
def dm02():
    # 创建numy数组
    n1 = np.array([11, 22, 33])
    print(f'n1: {n1}, type: {type(n1)}')

    # t1 = torch.from_numpy(n1)
    # t2 = t1.type(torch.float32)
    # 合并成下面这一行
    # t1 = torch.from_numpy(n1).type(torch.float32)
    t1 = torch.from_numpy(n1)   # 共享内存
    print(f't1: {t1}, type: {type(t1)}')
    # print(f't2: {t2}, type: {type(t2)}')

    t2 = torch.tensor((n1))     # 不共享内存
    print(f't2: {t2}, type: {type(t2)}')

    n1[0] = 100
    print(f'n1: {n1}')  # 【100, 22, 33]
    print(f't1: {t1}')  # 【100, 22, 33]
    print(f't2: {t2}')  # 【11, 22, 33]

# 场景3： 从标量张量中提取起内容
def dm03():
    # t1 = torch.tensor(100)
    t1 = torch.tensor([100, ])
    t2 = torch.tensor(True)
    # 张量里面只能是数值和布尔值，如果这里面写的是str类型的，就会报错
    print(f't1: {t1}, type: {type(t1)}')


    a = t1.item()
    b = t2.item()
    print(f'value: {a}, type: {type(a)}')

    print(f't2: {t2}, type: {type(t2)}')
    print(f'value: {b}, type: {type(b)}')


if __name__ == '__main__':
    dm03()