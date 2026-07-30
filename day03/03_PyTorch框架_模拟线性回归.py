import torch
from torch.utils.data import TensorDataset      # 构造数据集对象
from torch.utils.data import DataLoader         # 数据加载器
from torch import nn                            # nn 模块中有平方损失函数和假设函数
from torch import  optim                        # optim模块中有优化器函数
from sklearn.datasets import make_regression    # 创建线性回归模型数据集
import matplotlib.pyplot as plt                 # 可视化

# 设置绘图字体为黑体，中文不会变成方框
plt.rcParams['font.sans-serif'] = ['SimHei']     # 用来正常显示中文标签
# 坐标轴负号正常显示，不会变成方块
plt.rcParams['axes.unicode_minus'] = False       # 用来正常显示负号

# 定义函数，创建线性回归样本数据
def create_dataset():
    # 创建数据集对象
    x, y, coef = make_regression(   # coef就是权重w
        n_samples = 100,        # 100条样本（100个样本点）
        n_features = 1,         # 1个特征点
        noise = 10,             # 噪声，噪声越大，样本点越散，噪声越小样本点越集中
        coef = True,            # 是否返回系数（斜率），默认为False，返回值为None
        bias = 14.5,            # 偏置，真实直线的偏置b，真实公式 y=coef*x +14.5
        random_state = 3        # 随机种子，随机种子相同，输出数据相同，方便复现
    )
    # make_regression会生成x（特征）和y（目标值）。coef就是这条直线的斜率。
    # 真实规律是：y = coef * x + 14.5，但加上噪声后，点会围绕这条线上下波动。
    # print(type(x))   # 打印x的类型：这里是numpy数组，不能直接给PyTorch训练

    # 把上述的数据，封装乘张量对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 返回结果
    # 返回 特征x、标签y、真实权重coef
    return x, y, coef

# 定义函数，表示模型训练
def train(x, y, coef):
    # 1、创建数据集对象，把 tensor ——> 数据集对象 ——> 数据加载器
    dataset = TensorDataset(x, y)

    # 2、创建数据加载器对象
    # 参1：数据集对象，参2：批次大小，参3：是否打乱数据(训练集打乱，测试集不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # DataLoader：把数据分成一小批一小批（每批16个样本），并打乱顺序。这样每次训练时用一小批，可以加快速度，也让模型更稳定。

    # 3、创建初始的线性回归模型
    # 参1：输入的特征维度，参2：输出特征维度
    model = nn.Linear(1, 1)
    # nn.Linear(1, 1)创建一个线性层，输入特征维度 = 1，输出维度 = 1。
    # 这相当于创建了一个函数：y_pred = weight * x + bias。一开始，weight和bias是随机初始化的。

    # 4、创建损失函数对象
    criterion = nn.MSELoss()
    # MSE损失（均方误差）：计算预测值y_pred和真实值y的差距。值越小，说明预测越准。

    # 5、创建优化器对象
    # 参1：模型参数，参2：学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # SGD（随机梯度下降）：一种优化算法，负责调整模型的weight和bias，让损失变小。
    # lr = 0.01是学习率，控制每次调整的步伐大小。

    # 6、具体的训练步骤
    # 6.1 定义变量，分别表示：训练轮数，每轮的（平均）损失之，训练总损失值，训练的样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0

    # 6.2 开始训练，按轮训练
    for epoch in range(epochs):     # 外层循环：训练100轮 ,epoch的值：0, 1, 2, ..., 99
        # 6.3 每轮是分批次训练的，所以从数据加载器中获取批次数据
        for train_x, train_y in dataloader:     # 内层循环：每轮分批训练, 7批（16，16，16，16，16，16，4）
            # 6.4 模型预测
            y_pred = model(train_x)

            # 6.5 计算（每批的平均）损失值
            loss = criterion(y_pred, train_y.reshape(-1, 1))    # -1 自动计算

            # 6.6 计算总损失 和 样本（批次）数
            total_loss += loss.item()   # 累加当前批次损失到本轮总损失
            total_sample += 1           # 批次计数+1（一共7批）
            # 把每批的损失加起来，然后除以批次数，得到这一轮的平均损失。
            # 记录到loss_list，方便后面画图。

            # 6.7 标准训练三步：梯度清零 ——> 反向传播 ——> 梯度更新
            optimizer.zero_grad()   # 梯度清零：上一轮梯度清空，避免叠加干扰
            loss.backward()         # 反向传播：自动计算损失对w、b的梯度（误差往回传）
            optimizer.step()        # 梯度下降更新：根据梯度修改w和b，减小损失

        # 6.8 把本轮的（平均）损失之，添加到列表中【一轮所有批次跑完，计算本轮平均损失，存入列表用于画图】
        loss_list.append(total_loss / total_sample)
        # 打印当前轮数+本轮平均损失
        print(f'轮数：{epoch + 1}, 平均损失值：{total_loss / total_sample}')

    # 7、打印（最终的）训练结果
    print(f'{epochs}轮的平均损失分别为：{loss_list}')
    print(f'模型参数，权重：{model.weight}, 偏置：{model.bias}')

    # 8、绘制损失曲线
    #                  100轮     每轮的平均损失值
    plt.plot(range(epochs), loss_list)
    # 横轴是训练轮数，纵轴是损失值。如果曲线下降，说明模型在“学习”。

    plt.title('损失值曲线变化图')
    plt.grid()      # 绘制网格线
    plt.show()

    # 9、绘制预测值和真实值的关系
    # 9.1 绘制样本点分布情况
    plt.scatter(x, y)
    # plt.show()

    # 9.2 绘制训练模型的预测值
    # x: 100个样本点的特征
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])

    # 9.3 计算真实值
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])

    # 9.4 绘制预测值 和 真实值 的折线图
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.plot(x, y_true, color='green', label='预测值')

    # 9.5 图例，网格
    plt.legend()
    plt.grid()

    # 9.6 显示图像
    plt.show()

if __name__ == '__main__':
    # 创建数据集
    x, y, coef = create_dataset()
    # print(f'x: {x}, y: {y}, coef: {coef}')

    # 模型训练
    train(x, y, coef)