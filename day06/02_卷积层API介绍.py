import torch
import torch.nn as nn
import  matplotlib.pyplot as plt

# 定义函数，用于完成图像的加载，卷积，特征图可视化操作
def dm01():
    # 加载RGB真彩图
    img = plt.imread('./data/img.jpg')
    # 打印读取到的图像信息
    # print(f'img: {img}, shape: {img.shape}')    # HWC: (940, 940, 3)
    # 把图像的形状从 HWC ——> CHW，思路：img ——> 张量 ——> 转换维度
    img2 = torch.tensor(img, dtype=torch.float)
    img2 = img2.permute((2, 0, 1))
    print(f'img2: {img2}, shape: {img2.shape}')

    # 因为这里只有1张图，所以我们给它增加1个维度，从CHW——>(1, C, H, W), 1张3通道的 940*940像素的图
    img3 = img2.unsqueeze(dim=0)
    # unsqueeze(dim=N)：在第N维新增一个长度为1的维度，也就是给张量「升维、加一层括号」。
    # squeeze()删掉长度为1的维度。
    print(f'img3: {img3}, shape: {img3.shape}')

    # 创建卷积层对象，提取特征图
    # 参1：输入图像的通道数，参2：输出图像的通道数（几个特征图），参3：卷积核的大小，参4：步长，参5：填充
    conv = nn.Conv2d(3, 4, 3, 2, 0)

    # 具体的卷积计算
    conv_img = conv(img3)

    # 打印卷积后的结果 1张4通道【out_channel: 4】的 469*469 像素的图
    # print(f'conv_img: {conv_img}, shape: {conv_img.shape}') # (1, 4, 469, 469)

    # 查看提取的4个特征图
    img4 = conv_img[0]
    print(f'img4: {img4}, shape: {img4.shape}') # (4, 469, 469) —— CHW

    # 把上述的图从 CHW ——> HWC
    img5 = img4.permute(1, 2, 0)
    print(f'img5: {img5}, shape: {img5.shape}') # (469, 469, 4) —— HWC

    # 可视化第1个通道的特征图
    feature = img5[:, :, 0].detach().numpy()    #0通道(即第1通道的) (469, 469)像素图
    # 第一个:：所有高度像素（全部行）;第二个:：所有宽度像素（全部列）
    # 0：取第0个通道
    # 作用：把图片的第一个通道单独抽出来，得到单通道二维图[H, W]
    plt.imshow(feature)
    plt.show()

    feature1 = img5[:, :, 1].detach().numpy()
    plt.imshow(feature1)
    plt.show()

    feature2 = img5[:, :, 2].detach().numpy()
    plt.imshow(feature2)
    plt.show()


if __name__ == '__main__':
    dm01()