"""
计算机视觉模块torchvision自带的CIFAR10数据集
包含6万张(32, 32, 3)的图片，5万张训练集，1万张测试集，10个分类，每个分类6千张图片

"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from  torchvision.transforms import ToTensor
import torch.optim as optim
from  torch.utils.data import  DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

# 准备数据集
def create_dataset():
    # 获取训练集
    # 参1：数据集路径，参2：是否是训练集，参3：数据预处理——>张量数据，参4：是否联网下载
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)   # (50000, 32, 32, 3)
    # 获取测试集
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)   # (10000, 32, 32, 3)
    # 返回数据集
    return train_dataset, test_dataset

# 搭建(卷积)神经网络
class ImageModel(nn.Module):
    # 初始化父类成员，搭建神经网络
    def __init__(self):
        super().__init__()
        # 第一个卷积层
        # 输入 3 个通道，输出 6 个通道，卷积核大小3，步长1，填充0
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        # 窗口大小 2*2，步长2，填充0
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        # 第1个全连接层
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    # 定义前向传播
    def forward(self, x):
        # 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        # x = self.conv1(x)
        # x = torch.relu(x)
        # x = self.pool1(x)
        x = self.pool1(torch.relu(self.conv1(x)))

        # x = self.conv2(x)
        # x = torch.relu(x)
        # x = self.pool2(x)
        x = self.pool2(torch.relu(self.conv2(x)))

        # 拉平处理 (8, 16, 6, 6) ——> (8, 576)   # 可以直接用nn.Flatten
        # 参1：样本数（行数），参2：列数（特征数），-1表示自动计算
        x = x.reshape(x.size(0), -1)    # 8行576列
        # print(f'x.shape: {x.shape}')
        x = torch.relu((self.fc1(x)))
        x = torch.relu((self.fc2(x)))
        x = self.output(x)  # 后续用 多分类交叉损失函数 CrossEntropyLoss = Softmax() + 损失计算

        return x


# 模型训练
def train(train_dataset):
    # 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 创建模型对象
    model = ImageModel()
    # 创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 循环遍历epoch，开始每轮的训练动作
    epochs = 10     # 训练总轮数，10轮
    for epoch_idx in range(epochs):
        # 定义变量，记录3总损失，总样本数据量，预测正确样本个数，训练时间
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        # 遍历数据加载器，获取到每批次的数据
        for x, y in dataloader:
            # 切换训练模式
            model.train()
            # 模型预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 三步【梯度清零+反向传播+参数更新】
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计预测正确的样本个数
            # argmax()返回最大值对应的索引，充当——>该图片的预测分类
            # print(torch.argmax(y_pred, dim=-1))                 # -1代表行     预测分类
            # print(y)                                            # 真实分类
            # print(torch.argmax(y_pred, dim=-1) == y)            # 是否预测正确
            # print((torch.argmax(y_pred, dim=-1) == y).sum())    # 预测正确的样本个数
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()

            # 统计当前批次的总损失                    第1批平均损失 * 第1批样本个数
            total_loss += loss.item() * len(y)  # [第1批总损失 +第2批总损失+.....]
            # 统计当前批次的总样本个数
            total_samples += len(y)


        # 打印该轮的训练信息
        print(f'epoch: {epoch_idx + 1}, loss: {total_loss / total_samples:.5f}, acc: {total_correct / total_samples:.2f}, time: {time.time() - start:.2f}s')
        # break   # 这里写break，意味着只训练一轮

    # 保存模型
    torch.save(model.state_dict(), './model/image_model.pth')

# 模型测试
def evaluate(test_dataset):
    # 创建测试集 数据加载器
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 创建模型对象
    model = ImageModel()
    # 加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth'))
    # 定义变量，统计预测正确的样本个数，总样本个数
    total_correct, total_samples = 0, 0
    # 遍历数据加载器，获取到每批次的数据
    for x, y in dataloader:
        # 切换训练模式
        model.eval()
        # 模型预测
        y_pred = model(x)
        # 因为训练的时候用了CrossEntropyLoss，所以搭建神经网络时没有加softmax()激活函数，这里要用argmax()
        y_pred = torch.argmax(y_pred, dim=-1)
        # 统计预测正确的样本个数
        # print(torch.argmax(y_pred, dim=-1))
        total_correct += (y_pred == y).sum()
        # 统计总样本个数
        total_samples += len(y)

    # 打印正确率（预测结果）
    print(f'Acc: {total_correct / total_samples:.2f}')


if __name__ == '__main__':
    # # 获取数据集
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集：{train_dataset.data.shape}')
    # print(f'测试集：{test_dataset.data.shape}')
    # print(f'数据集类别：{train_dataset.class_to_idx}')
    # create_dataset()
    #
    # # 图像展示
    # # 创建画布，尺寸宽2英寸、高2英寸
    # plt.figure(figsize=(2, 2))
    # # 取出第12张图片（索引11，从0开始计数）
    # plt.imshow(train_dataset.data[11])   # 索引为11的图像
    # # 设置标题为图片对应的标签
    # plt.title(train_dataset.targets[11])
    # plt.show()

    # # 搭建神经网络
    # model = ImageModel()
    # # 查看模型参数
    # # 参1：模型，参2：输入维度(CHW，通道，高， 宽)，参3：批次大小
    # summary(model, (3, 32, 32), batch_size=BATCH_SIZE)


    # train(train_dataset)

    evaluate(test_dataset)