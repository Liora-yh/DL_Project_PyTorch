import torch
import re
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# 获取数据，进行分词，获取词表
def build_vocab():
    # 定义变量，记录：去重后所有的词，每行文本分词结果
    unique_words, all_words = [], []
    # 遍历数据集，获取到每行文本
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        # 获取到每行歌词，进行分词
        words = jieba.lcut(line)
        # print(f'每行数据：{words}')
        # 所有分词结果记录到 all_words 中
        all_words.append(words)
        # 遍历分词结果，去重后，添加到unique_wods中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        # 统计语料中（去重后）词的数量
    word_count = len(unique_words)      # 5703
    # print(word_count)
    # 构建词表，字典形式，key是词，value是次的索引
    word_to_index = {word:i for i, word in enumerate(unique_words)}
    # print(f'word_to_index: {word_to_index}')
    # 歌词文本用词表索引表示
    corpus_idx = []
    # 遍历每一行的分词结果
    for words in all_words:
        # print(words)
        # 定义变量，记录词索引列表
        tmp = []
        # 获取每一行的词，并获取相应的索引
        for word in words:
            tmp.append(word_to_index[word])
        # 在每行词直接，添加空格隔开
        tmp.append(word_to_index[' '])
        # 获取文档中每个词的索引，添加到corpus_idx中
        corpus_idx.extend(tmp)
        # print(f'corpus_idx: {corpus_idx}')
    # 返回结果：唯一此列表（5703个词）
    return unique_words, word_to_index, word_count, corpus_idx

# 数据预处理，构建数据集

# 搭建RNN神经网络

# 训练模型

# 模型预测

if __name__ == '__main__':
    # build_vocab()
    #获取数据，进行分词，获取词表
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    print(f'词的数量：{word_count}')         # 去重后，5703个词
    print(f'去重后的词：{unique_words}')
    print(f'每个词的索引：{word_to_index}')
    print(f'文档中每个词对应的索引：{corpus_idx}')
