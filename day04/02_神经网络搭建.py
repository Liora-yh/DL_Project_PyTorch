import torch
import torch.nn as nn
from torchsummary import summary

class ModelDemo(nn.Module):
    def __init__(self):

        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 搭建神经网络 ——> 隐藏层 + 输出层
        # 隐藏层1：输入特征数 3，输出特征数 3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2：输入特征数 3，输出特征数 2
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输入特征数 2，输出特征数 2
        self.output = nn.Linear(2, 2)

        # 1.3 对隐藏层进行参数初始化
        # 隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # 2.1 第一层 隐藏层计算：加权求和 + 激活函数(Sigmoid)
        # 分解版写法：
        # x = self.linear1(x)     # 加权求和
        # x = torch.sigmoid(x)    # 激活函数
        # 合并版写法
        x = torch.sigmoid(self.linear1(x))

        # 2.2 第二层 隐藏层计算：加权求和 + 激活函数(ReLU)
        x = torch.relu(self.linear2(x))

        # 2.3 第三层 输出层计算：加权求和 + 激活函数(softmax)
        # dim=-1, 表示按行计算；dim=0, 表示按列计算
        x = torch.softmax(self.output(x), dim=-1)

        # 2.4 返回预测值
        return x

# 模型预测
def train():
    # 创建模型对象
    my_model = ModelDemo()
    print(f'my_model: {my_model}')

    # 创建数据样本，随机生成
    data = torch.randn(size=(5, 3))
    print(f'data: {data}')
    print(f'data.shape: {data.shape}')      # 5行3列
    print(f'data.requires_grad: {data.requires_grad}')  # False

    # 调用神经网络模型 ——> 进行模型训练
    output = my_model(data)             # 底层自动调用了forward()方法
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')  # 5行2列
    print(f'output.requires_grad: {output.requires_grad}')  # True
    print('_' * 30)

    # 计算 和 查看模型参数
    print('============= 计算模型参数 ==============')
    # 参1：（神经网络）模型对象；参2：输入数据维度（5行3列）
    summary(my_model, input_size=(5, 3))

    print('============= 查看模型参数 ==============')
    for name, param in my_model.named_parameters():
        print(f'name: {name}')
        print(f'param: {param}\n')

if __name__ == '__main__':
    train()

