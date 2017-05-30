import re
import traceback
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import jieba
import numpy as np
from SVM import SVM

jieba.load_userdict("../train/word.txt")
stop = [line.strip() for line in open('../ad/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词


def build_key_word(path):  # 通过词频产生特征
    d = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            for word in jieba.cut(line.strip()):
                p = re.compile(r'\w', re.L)
                result = p.sub("", word)
                if not result or result == ' ':  # 空字符
                    continue
                if len(word) > 1:  # 避免大量无意义的词语进入统计范围
                    d[word] = d.get(word, 0) + 1
    kw_list = sorted(d, key=lambda x: d[x], reverse=True)
    size = int(len(kw_list) * 0.2)  # 取最前的30%
    mood = set(kw_list[:size])
    return list(mood - set(stop))


def loadDataSet(path):  # 返回每条微博的分词与标签
    line_cut = []
    label = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            temp = line.strip()
            try:
                sentence = temp[2:].lstrip()  # 每条微博
                first_label = int(temp[:2])
                if first_label == 3:
                    continue
                label.append(first_label)  # 获取标注
                word_list = []
                sentence = str(sentence).replace('\u200b', '')
                for word in jieba.cut(sentence.strip()):
                    p = re.compile(r'\w', re.L)
                    result = p.sub("", word)
                    if not result or result == ' ':  # 空字符
                        continue
                    word_list.append(word)
                word_list = list(set(word_list) - set(stop) - set('\u200b')
                                 - set(' ') - set('\u3000') - set('️'))
                line_cut.append(word_list)
            except Exception:
                continue
    return line_cut, label  # 返回每条微博的分词和标注


def setOfWordsToVecTor(vocabularyList, moodWords):  # 每条微博向量化
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in moodWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return np.array(vocabMarked)


def setOfWordsListToVecTor(vocabularyList, train_mood_array):  # 将所有微博准备向量化
    vocabMarkedList = []
    for i in range(len(train_mood_array)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, train_mood_array[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


def trainingNaiveBayes(train_mood_array, label):  # 计算先验概率
    numTrainDoc = len(train_mood_array)
    numWords = len(train_mood_array[0])
    prior_Pos, prior_Neg = 0.0, 0.0
    for i in label:
        if i == 1:
            prior_Pos = prior_Pos + 1
        elif i == 2:
            prior_Neg = prior_Neg + 1
    prior_Neg = prior_Neg / float(numTrainDoc)
    prior_Pos = prior_Pos / float(numTrainDoc)
    wordsInPosNum = np.ones(numWords)
    wordsInNegNum = np.ones(numWords)
    PosWordsNum = 2.0  # 如果一个概率为0，乘积为0，故初始化1，分母2
    NegWordsNum = 2.0
    for i in range(0, numTrainDoc):
        try:
            if label[i] == 1:
                wordsInPosNum += train_mood_array[i]
                PosWordsNum += sum(train_mood_array[i])  # 统计Pos中语料库中词汇出现的总次数
            elif label[i] == 2:
                wordsInNegNum += train_mood_array[i]
                NegWordsNum += sum(train_mood_array[i])
        except Exception as e:
            traceback.print_exc(e)
    pWordsPosicity = np.log(wordsInPosNum / PosWordsNum)
    pWordsNegy = np.log(wordsInNegNum / NegWordsNum)
    return pWordsPosicity, pWordsNegy, prior_Pos


def classify(pWordsPosicity, pWordsNegy, prior_Pos, test_word_arrayMarkedArray, D, DS_temp):
    pP = sum(test_word_arrayMarkedArray * pWordsPosicity * D) + np.log(prior_Pos)
    pN = sum(test_word_arrayMarkedArray * pWordsNegy * DS_temp) + np.log(1 - prior_Pos)
    if pP > pN:
        return pP, pN, 1
    else:
        return pP, pN, 2


def classify_without_D(pWordsPosicity, pWordsNegy, prior_Pos, test_word_arrayMarkedArray, D, DS_temp):
    pP = sum(test_word_arrayMarkedArray * pWordsPosicity ) + np.log(prior_Pos)
    pN = sum(test_word_arrayMarkedArray * pWordsNegy ) + np.log(1 - prior_Pos)
    if pP > pN:
        return pP, pN, 1
    else:
        return pP, pN, 2


def predict(pWordsPosicity, pWordsNegy, prior_Pos, test_word_arrayMarkedArray, D, DS_temp):
    errorCount = 0

    for j in range(testCount):
        try:
            pP, pN, smsType = classify_without_D(pWordsPosicity, pWordsNegy, prior_Pos, test_word_arrayMarkedArray[j], D, DS_temp)
            if smsType != test_word_arrayLabel[j]:
                errorCount += 1
        except Exception as e:
            traceback.print_exc(e)
    print(errorCount / testCount)


if __name__ == '__main__':
    for m in range(1, 11):
        vocabList = build_key_word("two_class.txt")
        line_cut, label = loadDataSet("two_class.txt")
        train_mood_array = setOfWordsListToVecTor(vocabList, line_cut)
        test_word_array = []
        test_word_arrayLabel = []
        testCount = 100  # 从中随机选取100条用来测试，并删除原来的位置
        for i in range(testCount):
            try:
                randomIndex = int(random.uniform(0, len(train_mood_array)))
                test_word_arrayLabel.append(label[randomIndex])
                test_word_array.append(train_mood_array[randomIndex])
                del (train_mood_array[randomIndex])
                del (label[randomIndex])
            except Exception as e:
                print(e)
        # 朴素贝叶斯
        # multi = MultinomialNB()
        # multi = multi.fit(train_mood_array, label)
        # joblib.dump(multi, 'model/gnb.model')
        # muljob = joblib.load('model/gnb.model')
        # result = muljob.predict(test_word_array)
        # count = 0
        # for i in range(len(test_word_array)):
        #     type = result[i]
        #     if type != test_word_arrayLabel[i]:
        #         count = count + 1
        # print("mul", count / float(testCount))
        # PosWords, NegWords, prior_Pos = trainingNaiveBayes(train_mood_array, label)
        # D = np.ones(len(vocabList))
        # DS_temp = D
        # predict(PosWords, NegWords, prior_Pos, test_word_array, D, DS_temp)
        # SVM
        model = SVM(max_iter=10, kernel_type='linear', C=1.0, epsilon=0.001)
        for i in range(len(label)):
            if label[i]==2:
                label[i]=-1
        for i in range(len(test_word_arrayLabel)):
            if test_word_arrayLabel[i]==2:
                test_word_arrayLabel[i]=-1
        model.fit(np.array(train_mood_array),np.array(label))
        y_hat = model.predict(np.array(test_word_array))
        num=0.0
        for i in range(len(y_hat)):
            if y_hat[i]!=test_word_arrayLabel[i]:
                num=num+1
        print(num/len(test_word_arrayLabel))