import torch

# 点乘
def dm01():
    t1 = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(f't1: {t1}')

    t2 = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(f't2: {t2}')

    # t3 = t1 * t2
    t3 = t1.mul(t2)
    print(f't3: {t3}')
    print('_' * 30)

def dm02():
    t1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
    print(f't1: {t1}')

    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(f't2: {t2}')

    t3 = t1 @ t2
    # t3 = t1.mat mul(t2)
    # t3 = t1.dot(t2)   # 报错，dot()只针对一维张量有效
    print(f't3: {t3}')
    print('_' * 30)

    t4 = torch.tensor([1, 2, 3])
    t5 = torch.tensor([4, 5, 6])
    t6 = t4.dot(t5)     # dot()只针对一维张量有效
    print(f't6: {t6}')

if __name__ == '__main__':
    # dm01()
    dm02()
