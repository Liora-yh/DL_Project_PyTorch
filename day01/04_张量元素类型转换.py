import torch

# 场景1： 直接创建指定类型的张量
t1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)   # 默认是float32
print(f't1: {t1}, (元素)类型：{t1.dtype}, (张量)类型：{type(t1)}')
data = torch.full([2, 3], 10)
print('_' * 30)

# 场景2：创建好张量后 ——> 做类型变换
# 思路1：ype()函数，推荐掌握
t2 = t1.type(torch.int16)
print(f't2: {t2}, (元素)类型：{t2.dtype}, (张量)类型：{type(t2)}')
print('_' * 30)

# 思路2：data.half()、data.double()、data.float()、data.short()等
print(t2.half())        # float16
print(t2.float())       # float32   关于小数，默认是float32
print(t2.double())      # float64
print(t2.short())       # int16
print(t2.int())         # int32
print(t2.long())        # int64     关于整数，默认是int64




print('_' * 30)
print(data)
print(data.dtype)


data = data.type(torch.DoubleTensor)
print(data.dtype)