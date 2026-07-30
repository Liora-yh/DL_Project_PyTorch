import torch

t1 = torch.tensor([1, 2, 3])

# 如果是张量和数值运算，则：该数值会和张量中的每个值依次进行对应的运算。
# t2 = t1.add(10)  # 不会修改原数据
# t2 = t1 + 10    # 不修改原数据

# t2 = t1.add_(10)   # 会修改原数据
# 或者直接写 t1.ad_(10)
# t1 += 10    # 会修改原数据

# sub()、mul()、div()、neg()
# t2 = t1.sub(1)
# t2 = t1.mul(2)
# t2 = t1.div(2)      # 或者写成 t2 = t1 / 2
# t2 = t1.neg()


t2 = t1 // 2    # //表示整除


print(f't1: {t1}')
print(f't2: {t2}')
