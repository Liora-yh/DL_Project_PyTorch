import torch
torch.manual_seed(24)

def dm01():
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1: {t1}, shape: {t1.shape}, row: {t1.shape[0]}, columns: {t1.shape[1]}, {t1.shape[-1]}')

    # 通过reshape()函数，把t1——>3行2列、1行6列
    # t2 = t1.reshape(3, 2)
    t2 = t1.reshape(1, 6)
    print(f't2: {t2}, shape: {t2.shape}, row: {t2.shape[0]}, columns: {t2.shape[1]}, {t2.shape[-1]}')

    # 通过reshape()函数，把t1——>2行5列
    t3 = t1.reshape(2, 5)   # 报错 转之前共计2*3=6个元素，转之后2*5=10个元素
    print(f't3: {t3}')


def dm02():
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1: {t1}, shape: {t1.shape}')   # (2, 3)

    # 在0维上添加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2: {t2}, shape: {t2.shape}')   # (1, 2, 3)

    # 在1维上添加一个维度
    t3 = t1.unsqueeze(1)
    print(f't2: {t3}, shape: {t3.shape}')   # (2, 1, 3)

    # 在2维上添加一个维度
    t4 = t1.unsqueeze(2)
    print(f't4: {t4}, shape: {t4.shape}')   # (2, 3, 1)

    # 在3维上(不存在)，添加一个维度      报错：越界
    # t5 = t1.unsqueeze(3)
    # print(f't5: {t5}, shape: {t5.shape}')   # (2, 3, *, 1)

    # 删除所有为1的维度
    t6 = torch.randint(1, 10, size=(2, 1, 3, 1, 1))
    print(f't6: {t6}, shape: {t6.shape}')      # (2, 1, 3, 1, 1)

    t7 = t6.squeeze()
    print(f't7: {t7}, shape: {t7.shape}')      # (2, 3)

def dm03():
    t1 = torch.randint(1, 10, size=(2, 3, 4))
    print(f't1: {t1}, shape: {t1.shape}')

    # 改变维度从(2, 3, 4) ——> (3, 2, 4)
    t2 = t1.transpose(0, 1)
    print(f't2: {t2}, shape: {t2.shape}')

    # 改变维度从(2, 3, 4) ——> (4, 3, 2)
    t3 = t1.transpose(0, 2)
    # 或者下面写法
    # t3 = t1.transpose(0, -1)
    print(f't3: {t3}, shape: {t3.shape}')

    # 改变维度从(2, 3, 4) ——> (4, 2, 3)
    t4 = t1.permute(2, 0, 1)
    print(f't4: {t4}, shape: {t4.shape}')


def dm04():
    t1 = torch.randint(1, 10, size=(2, 3))

    # 判断张量是否连续，即：张量的顺序与内存中的存储顺序是否一致
    print(t1.is_contiguous())   # True

    # 通过view()函数，修改上述张量的形状，从(2, 3) ——> (3, 2)
    t2 = t1.view(3, 2)
    print(f't2: {t2}, shape: {t2.shape}')
    print(t2.is_contiguous())   # True

    # 通过transpose()交换维度 ——> 交换之后，不连续了
    t3 = t1.transpose(0, 1)
    print(f't3: {t3}, shape: {t3.shape}')
    print(t3.is_contiguous())   # False

    # 通过contiguous()函数，把t3张量——>连续张量——>通过view()转成(2, 3)
    t4 = t3.contiguous().view(2, 3)
    print(f't4: {t4}, shape: {t4.shape}')


if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()
