import json
import numpy as np
import pandas as pd
from collections import Counter
import gensim

class Dataset(object):
    def __init__(self, config):
        self.config = config
#        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.seq_len # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.testReviews = []
        self.testLabels = []

        self.wordEmbedding = None

        self.labelList = []

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        df = pd.read_csv(filePath)

        if self.config.numClasses == 1 or self.config.numClasses == 2:
            labels = df["sentiment"].tolist()
        elif self.config.numClasses > 2:
            labels = df["rate"].tolist()

        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels

    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]
        labd =[]
        for lab in labelIds:
            if lab == 0:
                labd.append([0, 1])
            else:
                labd.append([1, 0])
        return labd

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

        trainReviews = np.asarray(reviews[:], dtype="int64")
        trainLabels = np.array(y[:], dtype="float32")

        return trainReviews, trainLabels

    def _genVocabulary(self, reviews, labels, vocab):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据

        with open("./data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)

        return label2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("./data/word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize).astype(np.float32))
        wordEmbedding.append(np.random.randn(self._embeddingSize).astype(np.float32))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        print("hah",np.shape(wordEmbedding))
        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化停用词
        self._readStopWord(self._stopWordSource)

        df = pd.read_csv("./data/preProcess/wordEmbdiing.csv")
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        # 初始化数据集
        trainReviews, trainLabels = self._readData("./data/preProcess/train.csv")
        train_label2idx = self._genVocabulary(trainReviews, trainLabels, vocab)
        train_labelIds = self._labelToIndex(trainLabels, train_label2idx)
        train_reviewIds = self._wordToIndex(trainReviews, word2idx)
        trainReviews, trainLabels = self._genTrainEvalData(train_reviewIds, train_labelIds, word2idx, self._rate)

        # 初始化数据集
        validReviews, validLabels = self._readData("./data/preProcess/valid.csv")
        valid_label2idx = self._genVocabulary(validReviews, validLabels, vocab)
        valid_labelIds = self._labelToIndex(validLabels, valid_label2idx)
        valid_reviewIds = self._wordToIndex(validReviews, word2idx)
        validReviews, validLabels = self._genTrainEvalData(valid_reviewIds, valid_labelIds, word2idx, self._rate)

        # 初始化数据集
        testReviews, testLabels = self._readData("./data/preProcess/test.csv")
        test_label2idx = self._genVocabulary(testReviews, testLabels, vocab)
        test_labelIds = self._labelToIndex(testLabels, test_label2idx)
        test_reviewIds = self._wordToIndex(testReviews, word2idx)
        testReviews, testLabels = self._genTrainEvalData(test_reviewIds, test_labelIds, word2idx, self._rate)


        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = validReviews
        self.evalLabels = validLabels

        self.testReviews = testReviews
        self.testLabels = testLabels

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY