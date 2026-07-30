import numpy as np
import matplotlib.pyplot as plt
import torch

# 绘制：全黑，全白图
def dm01():
    # 定义全黑图片：像素点越接近0越黑，越接近255越白
    # HWC:  H: 高度，W: 宽度，C: 通道
    img1 = np.zeros((200, 200, 3))
    print(f'img1: {img1}')

    # 绘制图片
    plt.imshow(img1)
    # plt.axis('off') # 关闭坐标系
    plt.show()

    # 定义全白图片
    img2 = torch.full(size=(200, 200, 3), fill_value=255)
    print(f'img2: {img2}')

    # 绘制图片
    plt.imshow(img2)
    plt.show()

# 加载图片
def dm02():
    # 加载图片
    img1 = plt.imread('data/img.jpg')
    print(f'img1: {img1}')
    print(f'img1.shape: {img1.shape}')

    # 保存图像
    plt.imsave('./data/img_copy.jpg', img1)

    # 展示图像
    plt.imshow(img1)
    plt.show()

if __name__ == '__main__':
    # dm01()
    dm02()