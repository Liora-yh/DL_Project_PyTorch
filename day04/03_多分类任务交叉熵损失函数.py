import torch
import torch.nn as nn

def dm01():
    # 手动创建样本的真实值 ——> 公式中的y
    # 独热编码（one-hot格式）
    # 第1个样本[0, 1, 0]：类别1为正确答案
    # 第2个样本[0, 0, 1]：类别2为正确答案
    # 规则：哪一列是1，代表样本属于第几类，其余位置填0。
    # y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=float)
    # 类别索引格式（标签索引）
    # 第1个样本真实类别编号：1
    # 第2个样本真实类别编号：2
    # 即样本1 → 类别1，样本2 → 类别2
    y_true = torch.tensor([1, 2])

    # nn.CrossEntropyLoss只能接收第二种索引格式，不能直接接收one-hot

    # 手动创建样本的真实值 ——> 公式中的 f(x)
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], requires_grad=True,dtype=torch.float)

    # 创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()       # 平均损失，来源于参数：reduction: str = "mean"

    # 计算损失值
    loss = criterion(y_pred, y_true)
    print(f'损失值：{loss}')

if __name__ == '__main__':
    dm01()