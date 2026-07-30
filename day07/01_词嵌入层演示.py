import torch
import torch.nn as nn
import jieba    # jieba分词器，需要安装一下，pip install jieba

def dm01():
    # 定义一句话
    text = '北京冬奥的进度条已经过半，不少外国运动员再完成自己的比赛后踏上归途。'
    # 使用jieba模块进行分词
    words = jieba.lcut(text)
    print(f'分词结果：{words}')

    # 创建词嵌入层
    # 参1：词表大小（词的个数），参2：词向量的维度
    embed = nn.Embedding(len(words), 4)

    # 获取每个词对象的下标索引
    # i = 0
    # for word in words:
    #     print(i, word)
    #     i += 1
    # enumerate(): 返回列表中每个值及其对应的索引
    for i, word in enumerate(words):    # 效果同上
        # print(i, word)

        # 把词索引(张量形式)转成词向量
        word_vector = embed(torch.tensor(i))    # 随机的，每次都不一样
        print(f'词：{word}, \t\t词向量：{word_vector}')


if __name__ == '__main__':
    dm01()
