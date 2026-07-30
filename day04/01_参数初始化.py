import torch.nn as nn
from torch.nn.modules import linear


# 1、均匀分布随机初始化
def dm01():
    # 创建1个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 对权重（w）进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    # 对偏置（b）进行随机初始化
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

# 2、固定初始化
def dm02():
    # 创建1个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 对权重（w）进行随机初始化，设置固定值为3
    nn.init.constant_(linear.weight, 3)
    # 对偏置（b）进行随机初始化
    nn.init.constant_(linear.bias, 3)
    print(linear.weight.data)
    print(linear.bias.data)
# 3、全0初始化
def dm03():
    # 创建1个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 对权重（w）进行初始化
    nn.init.zeros_(linear.weight)
    # 对偏置（b）进行初始化
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)
# 4、全1初始化
def dm04():
    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight.data)
# 5、正态分布随机初始化
def dm05():
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight)
    print(linear.weight.data)
# 6、kaiming 初始化
def dm06():
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight)
    # nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight.data)
# 7、xavier初始化
def dm07():
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    # nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)

if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    dm07()
