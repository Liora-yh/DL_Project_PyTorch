import  torch

# 1、创建线性张量
def dm01():
    # 场景1：创建指定范围的线性张量
    # 参数1：起始值，参数2：结束值，参数3：步长
    t1 = torch.arange(0, 10, 2)  # 0, 2, 4, 6, 8
    print(f't1: {t1}, type: {type(t1)}')
    print('_' * 30)

    # 场景2：创建指定范围的线性张量 ————> 等差数列
    # 参数1：起始值，参数2：结束值，参数3：元素的个数
    t2 = torch.linspace(1, 10, 5)
    print(f't1: {t2}, type: {type(t2)}')
    print('_' * 30)

# 2、创建随机张量
def dm02():
    # step1: 设置随机种子
    # torch.initial_seed()    # 默认采用当前系统的时间戳做随机种子
    torch.manual_seed(3)    # 设置随机种子

    # step2: 创建随机张量
    # 场景1：均匀分布的（0， 1）随机变量
    t1 = torch.rand(size=(2, 3))       # 创建一个2行3列的随机矩阵
    print(f't1: {t1}, type: {type(t1)}')
    print('_' * 30)

    # 场景2：符合正态分布的随机变量
    t2 = torch.randn(size=(2, 3))
    print(f't1: {t2}, type: {type(t2)}')
    print('_' * 30)

    # 场景3： 创建随机证书张量
    t3 = torch.randint(low=1, high=10, size=(3, 5))
    print(f't1: {t3}, type: {type(t3)}')

if __name__ == '__main__':
    # dm01()
    dm02()