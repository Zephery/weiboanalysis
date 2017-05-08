# coding:utf-8


import json
from sklearn.externals import joblib
import numpy as np
import numpy.linalg as la

POSITIVE = 1  # 正面词语
NEGATIVE = 2  # 反面词语
NEUTRAL = 3  # 客观词语


def load_key_words(file_path):
    with open(file_path, encoding="utf-8") as fp:
        lines = fp.readlines()
        lines = [line.replace("\n", "") for line in lines]
    return lines


class DateSet:
    def __init__(self, data, label):
        self.data = np.array(data)
        self.label = label

    def Data(self):
        return self.data

    def Label(self):
        return self.label


def load_date_sets(file_path):
    with open(file_path, encoding="utf-8") as f:
        data_list = json.load(f)
    temp = []
    for data in data_list:
        # 前面是特征向量，后面最后一个是标签
        label = data[-1]
        feature = data[:-1]
        d = DateSet(feature, label)
        temp.append(d)
    return temp


# 欧式距离,1表示100%，越接近0表示越不相似
def _ecl_sim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# 皮尔逊相关系数,范围-1->+1， 越大越相似
def _pears_sim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦相关范围-1->+1 越大越相似
def _cos_sim(inA, inB):
    num = float(inB * inA.T)
    de_nom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / de_nom)


def _get_feature(sentence, key_word):
    size = len(key_word)
    feature = [0 for _ in range(size)]
    for index in range(size):
        word = key_word[index]
        value = sentence.find(word)  # 单词最初出现的位置
        if value != -1:
            feature[index] = 1
    return np.array(feature)


def get_mood(sentence, key_word, model_name):
    feature = _get_feature(sentence, key_word)
    gnb = joblib.load(model_name)
    pre_y = gnb.predict([feature])
    result = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }
    try:
        if pre_y[0] == POSITIVE:
            result["positive"] = 1
        elif pre_y[0] == NEGATIVE:
            result["negative"] = 1
        elif pre_y[0] == NEUTRAL:
            result["neutral"] = 1
    except:
        pass
    return result
