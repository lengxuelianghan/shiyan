import pandas as pd
import csv
from bs4 import BeautifulSoup

# unlabeledTrain = []
# with open("./preProcess/test.csv", "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         unlabeledTrain.append(row)
#
# unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])
#
# print(unlabel)
# def cleanReview(subject):
#     # 数据处理函数
#     beau = BeautifulSoup(subject)
#     newSubject = beau.get_text()
#     newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '')\
#         .replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')\
#         .replace('`', '').replace('!', '').replace(':', '')
#     newSubject = newSubject.strip().split(" ")
#     newSubject = [word.lower() for word in newSubject]
#     newSubject = " ".join(newSubject)
#
#     return newSubject
#
# unlabel["review"] = unlabel["review"].apply(cleanReview)
#
# unlabel.to_csv("./test.csv", index=False)


# unlabeledTrain = []
# with open("./preProcess/test.csv", "r") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         unlabeledTrain.append(row)
#
# unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])
#
# print(unlabel)
# def cleanReview(subject):
#     # 数据处理函数
#     beau = BeautifulSoup(subject)
#     newSubject = beau.get_text()
#     newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '')\
#         .replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')\
#         .replace('`', '').replace('!', '').replace(':', '')
#     newSubject = newSubject.strip().split(" ")
#     newSubject = [word.lower() for word in newSubject]
#     newSubject = " ".join(newSubject)
#
#     return newSubject
#
# unlabel["review"] = unlabel["review"].apply(cleanReview)
#
# unlabel.to_csv("./test.csv", index=False)


with open("./preProcess/datasetSentences.txt", "r") as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

unlabeledTrain_=[]
for i in range(len(unlabeledTrain) - 1):
    if i > 0:
        unlabeledTrain_.append(unlabeledTrain[i][1])

unlabel = pd.DataFrame(unlabeledTrain_, columns=["review"])
print(unlabel.head(5))

def cleanReview(subject):
    # 数据处理函数
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '')\
        .replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')\
        .replace('`', '').replace('!', '').replace(':', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject

unlabel["review"] = unlabel["review"].apply(cleanReview)
print(unlabel.head(5))

# label["review"] = label["review"].apply(cleanReview)

# 将有标签的数据和无标签的数据合并
#newDf = pd.concat([unlabel["review"], label["review"]], axis=0)
# 保存成txt文件
unlabel.to_csv("./preProcess/wordEmbdiing.csv", index=False)
#
# with open("./wordEmbdiing.txt", "r", encoding="utf-8") as f:
#      unlabeledTrain = [line.strip()for line in f.readlines()]
# k = 0
# for i in range(len(unlabeledTrain)):
#     print(len(unlabeledTrain[i]))
#     k += len(unlabeledTrain[i])
# kk = k // len(unlabeledTrain)
# print("kk",kk)
#
#
import logging
import gensim
from gensim.models import word2vec

# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("./wordEmbdiing.txt")

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)
model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True)

# 加载bin格式的模型
wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)