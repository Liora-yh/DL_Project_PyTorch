"""
    ANN (人工神经网络)案例：手机价格分类

    基于手机的20列特征 ——> 预测手机的加个区间(4个区间)

ANN案例的实现步骤：
    1、构建数据集
    2、搭建神经网络
    3、模型训练
    4、模型测试
"""
import torch            # PyTorch框架，封装了张量的各种操作
import torch.nn as nn           # neural network,，封装了神经网络的各种操作
from torch.utils.data import  TensorDataset     # 数据集对象，数据 ——> Tensor ——> 数据集 ——> 数据加载器
from torch.utils.data import  DataLoader        # 数据加载器
import torch.optim as optim                     # 优化器
# from sklearn.datasets import  make_regression
from sklearn.model_selection import  train_test_split       # 训练集和测试集的划分
import matplotlib.pyplot as plt     # 绘图
import  numpy as np                 # 数组（矩阵）操作
import  pandas as pd                # 数据处理
import  time                        # 时间模块
from torchsummary import  summary   # 模型结构可视化

# 1、定义函数，构架数据集
def create_dataset():
    # 加载csv文件数据集
    data = pd.read_csv('./data/train.csv')
    # print(f'data: {data.head()}')     # .head()：打印前5行数据，快速看一眼数据长什么样
    # print(f'data: {data.shape}')

    # 获取x特征列和y标签列
    # iloc[行切片, 列切片]，按数字下标取行和列
    # :：所有行全部保留
    # :-1：列从第0列到倒数第二列，去掉最后一列
    # -1：只取最后一列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    print(f'x: {x.head()}, {x.shape}')
    print(f'y: {y.head()}, {y.shape}')

    # 把特征列转成浮点型
    x = x.astype((np.float32))
    # astype()是pandas表格 / NumPy数组自带的数据类型转换方法，作用：把整张表 / 一列数据统一改成你指定的数据类型
    print(f'x: {x.head()}, {x.shape}')

    # 切分训练集和测试集
    # 参1：特征，参2：标签，参3：测试集所占比例，参4：随机种子，参5：样本的分布（即参考y的类别进行抽取数据）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)
    # 前面拆分好的全部特征、全部标签，一起传入，保证特征和标签一一对应不打乱
    # test_size = 0.2——测试集占总数据的比例：0.2 = 20 %，剩下80 % 自动作为训练集。
    # stratify=y 核心作用：分层抽样
    # stratify 翻译：分层，以传入的y标签为基准划分数据集。
    # 不加stratify = y：纯随机拆分，有可能测试集全是类别0，类别1一条都没有，数据分布失衡，评估不准
    # 加stratify = y：强制保证训练集、测试集内部各类别的占比，和原始全集y完全一致

    # 把数据集封装成张量数据集，数据 ——> 张量Tensor ——> 数据集TensorDataSet ——> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    # 返回结果                             20(充当输入特征数)   4(充当输出标签数)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))
    # .shape[0] = 样本总数
    # .shape[1] = 单个样本有多少个特征
    # np.unique(y)提取标签 y 里所有不重复的类别；len(...)统计不重复类别的总数量 = 分类任务的总类别数

# 搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)
    def forward(self, x):
        # 隐藏层1：加权就和 + 激活函数(relu)
        # x = self.linear1(x)
        # x = torch.relu(x)
        x = torch.relu(self.linear1(x))
        # 隐藏层2：加权就和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))
        # 输出层：加权求和 + 激活函数(softmax) ——> 这里只需要做加权求和
        # 正常写法，但是不需要，后续用多分类交叉熵损失函数CrossEntropLoss()替代
        # CrossEntropyLoss() = softmax() + 损失计算
        # x = torch.softmax(self.output(x), dim=1)
        x = self.output(x)
        return x
# 模型训练
def train(train_dataset, input_dim, output_dim):
    # 创建数据加载器
    # 参1：数据集对象（1600条），参2：每批次的数据条数，参2：是否打乱数据（训练集L打乱，测试集：不打乱）
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 创建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)
    # 定义损失函数，因为是多分类，这里用是：多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 模型训练
    # 定义变量，记录训练的总轮数
    epochs = 50
    # 开始每轮的训练
    for epoch in range(epochs):
        # 定义变量，记录每次训练的损失值，训练的批次数
        total_loss, batch_num = 0.0, 0
        # 定义变量，表示训练开始的时间
        start = time.time()
        # 开始本轮的各个批次的训练
        for x, y in train_loader:
            # 切换模型（状态）
            model.train()   # 训练模式       model.eval() ——> 测试模式
            # 模型预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 梯度清零，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失值
            total_loss += loss.item()   # 把本轮的每批次（16条）的平均损失累计起来
            batch_num += 1
        # 至此本轮训练结束，打印训练信息
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}')
        # :.4f 和 :.2f 是f-string 格式化小数，控制数字保留几位小数：
        # .4f：f = float 浮点数，保留小数点后 4 位
        # .2f：保留小数点后 2 位
    # 多轮训练结束，保存模型（参数）
    # 参1：模型对象的参数*权重矩阵，偏置矩阵），参2：模型保存的文件名
    print(f'\n\n模型的参数信息：{model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/phone.pth') # 后缀名用：pth、pkl、pickle均可


# 模型测试
def evaluate(test_dataset, input_dim, output_dim):
    # 创建神经网络分类对象
    model = PhonePriceModel(input_dim, output_dim)
    # 加载模型参数
    model.load_state_dict(torch.load('./model/phone.pth'))
    # 创建测试集的数据加载起对象
    # 参1：数据集对象（400条），参2：每批次的数据条数，参2：是否打乱数据（训练集L打乱，测试集：不打乱）
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 定义变量，记录预测正确的样本个数
    correct = 0
    # 从数据加载器中，活到每批次的数据
    for x, y in test_loader:
        # 切换模型状态
        model.eval()
        # 模型预测
        y_pred = model(x)
        print(f'y_pred: {y_pred}')
        # 根据加权就和，得到类别，用argmax()获取最大值对应的下标，就是类别
        y_pred = torch.argmax(y_pred, dim=1)    # dim=1表示逐行处理，dim=0逐列处理
        # print(f'y_pred: {y_pred}')
        # print(f'y: {y}')
        # 统计预测正确的样本个数
        # print(y_pred == y)
        # print((y_pred == y).sum())
        correct += (y_pred == y).sum()

    # 打印准确率
    print(f'准确率(Accuracy): {correct / len(test_dataset):.4f}')

if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'训练集 数据集对象：{train_dataset}')
    print(f'测试集 数据集对象：{test_dataset}')
    print(f'输入特征数：{input_dim}')     # 20
    print(f'输出标签数：{output_dim}')    # 4

    model = PhonePriceModel(input_dim, output_dim)
    # 参1：模型对象，参2：输入数据的形状（批次大小，输入特征数），每批16条，每条20列特征
    summary(model, input_size=(16, input_dim))

    train(train_dataset, input_dim, output_dim)

    evaluate(test_dataset, input_dim, output_dim)