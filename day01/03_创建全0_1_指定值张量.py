import  torch

#===================== 场景1： torch.one和torch.ones_like创建全1张量
t1 = torch.ones(2, 3)       # 创建2行3列的全1张量
print(f't1: {t1}, type: {type(t1)}')
print('_' * 30)

# t2: 3行2列
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2: {t2}, type: {type(t2)}')

# t3 ————> 基于t2的形状，创建全1张量
t3 = torch.ones_like(t2)
print(f't3: {t3}, type: {type(t3)}')   # 3行2列 ————> 全1矩阵
print('*' * 30)

#================= 场景2： torch.zeros和torch.zeros_like创建全0张量
t1 = torch.zeros(2, 3)       # 创建2行3列的全0张量
print(f't1: {t1}, type: {type(t1)}')
print('_' * 30)

# t2: 3行2列
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2: {t2}, type: {type(t2)}')

# t3 ————> 基于t2的形状，创建全0张量
t3 = torch.zeros_like(t2)
print(f't3: {t3}, type: {type(t3)}')   # 3行2列 ————> 全0矩阵
print('*' * 30)

#================ 场景3： torch.full和torch.full_like创建全为指定值张量
t1 = torch.full(size=(2, 3), fill_value=255)       # 创建2行3列的全255张量
print(f't1: {t1}, type: {type(t1)}')
print('_' * 30)

# t2: 3行2列
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2: {t2}, type: {type(t2)}')

# t3 ————> 基于t2的形状，创建全255张量
t3 = torch.full_like(t2, 255)
print(f't3: {t3}, type: {type(t3)}')   # 3行2列 ————> 全0矩阵
print('*' * 30)