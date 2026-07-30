import torch
torch.manual_seed(24)

t1 = torch.randint(1, 10, (2, 3))
print(f't1: {t1}, shape: {t1.shape}')

t2 = torch.randint(1, 10, (2, 3))
print(f't2: {t2}, shape: {t2.shape}')

# cat()函数拼接张量
# 除了拼接的那个维度外，其他维度必须保持一样
# 0维拼接
t3 = torch.cat([t1, t2], dim=0)     # (2, 3) + (2, 3) = (4, 3)
print(f't3: {t3}, shape: {t3.shape}')
# 1维拼接
t4 = torch.cat([t1, t2], dim=1)     # (2, 3) + (2, 3) = (2, 6)
print(f't4: {t4}, shape: {t4.shape}')

t5 = torch.cat([t1, t2], dim=-1)
print(f't5: {t5}, shape: {t5.shape}')

print('_' * 30)

# stack()拼接张量，可以是新维度，无论是新旧维度，所有维度都必须保持一致
# 0维拼接
t6 = torch.stack([t1, t2], dim=0)   # (2, 3) + (2, 3) = (2, 2, 3)
print(f't6: {t6}, shape: {t6.shape}')

# 1维拼接
t7 = torch.stack([t1, t2], dim=1)   # (2, 3) + (2, 3) = (2, 2, 3)
print(f't7: {t7}, shape: {t7.shape}')

t8 = torch.stack([t1, t2], dim=2)   # (2, 3) + (2, 3) = (2, 3, 2)
print(f't8: {t8}, shape: {t8.shape}')