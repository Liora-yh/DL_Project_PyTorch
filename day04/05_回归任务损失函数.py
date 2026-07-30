import torch
import torch.nn as nn

# MAE损失函数【L1Loss】
def dm01():
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    criterion = nn.L1Loss()

    loss = criterion(y_pred, y_true)
    print(f'MAE损失值：{loss}')

# MSELoss
def dm02():
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    criterion = nn.MSELoss()

    loss = criterion(y_pred, y_true)
    print(f'MSE损失值：{loss}')


# Smooth L1
def dm03():
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    criterion = nn.SmoothL1Loss()

    loss = criterion(y_pred, y_true)
    print(f'Smooth L1损失值：{loss}')

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
