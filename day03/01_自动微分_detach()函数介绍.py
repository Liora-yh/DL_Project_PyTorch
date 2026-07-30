"""
    一个张量一旦设置了自动微分，这个张量就不能直接转成 numpy 的 ndarray 对象了，
    需要通过 detach()函数解决
"""
import torch
import numpy as np

t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)
print(f't1: {t1}, type: {type(t1)}')

# 通过 detach() 函数，拷贝一份张量，然后转换成 numpy 对象
t2 = t1.detach()
print(f't2: {t2}, type: {type(t2)}')

# 测试上述的 t1 和 t2 是否共享同一块空间=====共享
t1.data[0] = 100
print(f't1: {t1}, type: {type(t1)}')
print(f't2:c{t2}, type: {type(t2)}')
print('_' * 30)

# 查看 t1 和 t2 谁可以自动微分
print(f't1: {t1.requires_grad}')    # Ture
print(f't2: {t2.requires_grad}')    # False
print('_' * 30)

# 把 t2 转成 numpy 对象
n1 = t2.numpy()
print(f'n1:{n1}, type: {type(n1)}')

# 最终版
n2 = t1.detach().numpy()
print(f'n2:{n2}, type: {type(n2)}')
