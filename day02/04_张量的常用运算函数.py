import torch
def dm01():
    t1 = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(f't1: {t1}')

    print(t1.sum(dim=0))    # 按列求和
    print(t1.sum(dim=1))    # 按行求和
    print(t1.sum())    # 整体求和
    print('_' * 30)

    print(t1.max(dim=0))  # 按列求最大值
    print(t1.max(dim=1))  # 按行求最大值
    print(t1.max())       # 整体求最大值
    print('_' * 30)

    t2 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]], dtype=torch.float)
    print(t2.mean(dim=0))  # 按列求平均值
    print(t2.mean(dim=1))  # 按行求平均值
    print(t2.mean())       # 整体求平均值
    print('_' * 30)

    print(t1.pow(2))  # 平方
    print(t1.pow(3))  # 立方
    print(t1 ** 3)    # 立方
    print('_' * 30)

    print(t1.sqrt())  # 平方根
    print('_' * 30)

    print(t1.exp())  # e^n， n就是矩阵中的每个元素
    # e^1, e^2, e^3, e^4, e^5, e^6
    print('_' * 30)

    print(t2.log())
    print('_' * 30)

if __name__ == '__main__':
    dm01()