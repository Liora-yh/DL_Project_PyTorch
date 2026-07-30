import torch
import torch.nn as nn

def dm01():
    # 设置真实值
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)

    # 设置预测值（概率）
    y_pred = torch.tensor([0.6901, 0.5432, 0.2632])

    # 创建二分类交叉熵损失函数
    criterion = nn.BCELoss()

    # 计算损失
    loss = criterion(y_pred, y_true)
    print(f'损失值：{loss}')

if __name__ == '__main__':
    dm01()