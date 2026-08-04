import torch
import re
import jieba
from sympy.matrices.expressions import transpose
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
# 定义数据集类，继承 torch.utils.data.Dataset
class LyricsDataset(torch.utils.data.Dataset):
    # 初始化词索引，词个数等...
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中词的索引
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数
        self.num_chars = num_chars
        # 文档数据中词的数量，不去重，100000个词
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count // self.num_chars
    # 当使用len(obj)时，自动调用此方法
    def __len__(self):
        return self.number

    # 当使用obj[index]时，自动调用此方法
    def __getitem__(self, idx):
        # idx: 指向是词的索引，并将其修正索引值到文档的范围里面
        # 确保索引start再合法范围内，避免越界，start: 当前样本的起始索引
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)

        # 计算当前样本的结束索引
        end = start + self.num_chars

        # 输入值，从文档中取出start到end的索引值，作为x
        x = self.corpus_idx[start:end]      # [0:5]、[1:6]、...

        # 输出值，网络预测结果
        y = self.corpus_idx[start+1:end+1]

        # 返回输入值和输出值——>张量形式(x, y)
        return torch.tensor(x), torch.tensor(y)

# 搭建RNN神经网络
class TextGenerator(nn.Module):
    # 初始化方法
    def __init__(self,unique_word_count):   # unique_word_count: 去重的词的数量(703)
        # 初始化父类的成员
        super().__init__()
        # 初始化词嵌入层：语料中词的数量，词向量的维度
        self.ebd = nn.Embedding(unique_word_count,128)
        # 循环网络层：词向量维度，隐藏层维度：256，网络层：1
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层（全连接层）：特征向量维度（和隐藏层向量维度一致），词表中词的个数
        self.out = nn.Linear(256, unique_word_count)   # 词表中每个词的概率 ——> 选概率最大的那个词作为预测结果

    # 前向传播方法
    def forward(self, inputs, hidden):
        # 初始化 嵌入层处理
        # embd格式：(batch句子的数量，句子的长度，词向量维度)
        embd = self.ebd(inputs)
        # print(f'嵌入层处理后的维度：{embd.shape}')
        # rnn 处理
        # rnn格式：(句子的长度，batch句子的数量，隐藏层维度)
        output, hidden = self.rnn(embd, transpose(0, 1), hidden)
        # print(f'循环网络层处理后的维度：{output.shape}')
        # print(f'隐藏层处理后的维度：{hidden.shape}')
        # 全连接，输入内容必须是二维数据，即：词的数量 * 次的维度
        # output = self.out(output.transpose(0, 1).reshape(-1, 256))
        # 输入维度：(seq_len句子数量 * batch，词向量维度256)
        # 输出维度：(seq_len句子数量 * batch，词表中次的个数)
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))  # 同上
        # 返回结果,预测结果，隐藏层
        return output, hidden

    # 隐藏层的初始化方法
    def init_hidden(self, bs):      # batch_size
        # 隐藏层初始化：[网络层数，batch，隐藏层向量维度]
        return torch.zero(1, bs, 256)

# 训练模型

# 模型预测

if __name__ == '__main__':
    # build_vocab()
    # 1、获取数据，进行分词，获取词表
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print(f'词的数量：{word_count}')         # 去重后，5703个词
    # print(f'去重后的词：{unique_words}')
    # print(f'每个词的索引：{word_to_index}')
    # print(f'文档中每个词对应的索引：{corpus_idx}')

    # # 2、构建数据集
    # dataset = LyricsDataset(corpus_idx, 5)
    # print(f'句子数量：{len(dataset)}')
    # # 查看输入值和目标值
    # x, y = dataset[1]
    # print(f'输入值：{x}')
    # print(f'目标值：{y}')

    # 3、创建模型对象
    model = TextGenerator(word_count)
    # 查看参数
    for name, parameter in model.named_parameters():
        print(f'参数名称：{name}, 参数维度：{parameter.shape}')
