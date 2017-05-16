#!/usr/bin/python2.7
import re
import traceback
import numpy as np


def textParser(text):
    """
    对SMS预处理，去除空字符串，并统一小写
    :param text:
    :return:
    """
    regEx = re.compile(r'[^a-zA-Z]|\d')  # 匹配非字母或者数字，即去掉非字母非数字，只留下单词
    words = regEx.split(text)
    # 去除空字符串，并统一小写
    words = [word.lower() for word in words if len(word) > 0]
    return words


def loadSMSData(fileName):
    """
    加载SMS数据
    :param fileName:
    :return:
    """
    f = open(fileName)
    classCategory = []  # 类别标签，1表示是垃圾SMS，0表示正常SMS
    moodWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            classCategory.append(0)
        elif linedatas[0] == 'Pos':
            classCategory.append(1)
        # 切分文本
        words = textParser(linedatas[1])
        moodWords.append(words)
    return moodWords, classCategory


def createVocabularyList(moodWords):
    """
    创建语料库
    :param moodWords:
    :return:
    """
    vocabularySet = set([])
    for words in moodWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList


def getVocabularyList(fileName):
    """
    从词汇列表文件中获取语料库
    :param fileName:
    :return:
    """
    fr = open(fileName)
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList


def setOfWordsToVecTor(vocabularyList, moodWords):
    """
    SMS内容匹配预料库，标记预料库的词汇出现的次数
    :param vocabularyList:
    :param moodWords:
    :return:
    """
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in moodWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return np.array(vocabMarked)


def setOfWordsListToVecTor(vocabularyList, moodWordsList):
    """
    将文本数据的二维数组标记
    :param vocabularyList:
    :param moodWordsList:
    :return:
    """
    vocabMarkedList = []
    for i in range(len(moodWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, moodWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


def trainingNaiveBayes(trainMarkedWords, trainCategory):
    """
    训练数据集中获取语料库中词汇的
    Pos：P（Wi|Pos总词数）
    Neg：P（Wi|Neg总词数）
    Neutral：P（Wi|Neutral总词数）
    """
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])
    # 是积极的先验概率P(S)
    pPos, pNeg, pNeutral = 0.0, 0.0, 0.0
    for i in trainCategory:
        if i == 1:
            pPos = pPos + 1
        elif i == 2:
            pNeg = pNeg + 1
        else:
            pNeutral = pNeutral + 1
    pPos = pPos / float(numTrainDoc)
    pNeg = pNeg / float(numTrainDoc)
    pNeutral = pNeutral / float(numTrainDoc)
    wordsInPosNum = np.ones(numWords)
    wordsInNegNum = np.ones(numWords)
    wordsInNeutralNum = np.ones(numWords)
    PosWordsNum = 2.0
    NegWordsNum = 2.0
    NeutralWordsNum = 2.0
    for i in range(0, numTrainDoc):
        try:
            if trainCategory[i] == 1:  # 如果是垃圾SMS或邮件
                wordsInPosNum += trainMarkedWords[i]
                PosWordsNum += sum(trainMarkedWords[i])  # 统计Pos中语料库中词汇出现的总次数
            elif trainCategory[i] == 2:
                wordsInNegNum += trainMarkedWords[i]
                NegWordsNum += sum(trainMarkedWords[i])
            else:
                wordsInNeutralNum += trainMarkedWords[i]
                NeutralWordsNum += sum(trainMarkedWords[i])
        except Exception as e:
            traceback.print_exc(e)
    pWordsPosicity = np.log(wordsInPosNum / PosWordsNum)
    pWordsNegy = np.log(wordsInNegNum / NegWordsNum)
    pWordsNeutral = np.log(wordsInNeutralNum / NeutralWordsNum)
    return pWordsPosicity, pWordsNegy, pWordsNeutral, pPos, pNeg, pNeutral


def getTrainedModelInfo():
    """
    获取训练的模型信息
    :return:
    """
    # 加载训练获取的语料库信息
    vocabularyList = getVocabularyList('vocabularyList.txt')
    pWordsNegy = np.loadtxt('pWordsNegy.txt', delimiter='\t')
    pWordsPosicity = np.loadtxt('pWordsPosicity.txt', delimiter='\t')
    fr = open('pPos.txt')
    pPos = float(fr.readline().strip())
    fr.close()

    return vocabularyList, pWordsPosicity, pWordsNegy, pPos


def classify(pWordsPosicity, pWordsNegy, pWordsNeutral, DS, pPos, pNeg, pNeutral, testWordsMarkedArray, DS_neg,
             DS_neutral):
    """
    计算联合概率进行分类
    :param DS:  adaboost算法额外增加的权重系数
    np.log(x)返回x的自然对数    即：ln(x)
    """
    # 计算P(Ci|W)，W为向量。P(Ci|W)只需计算P(W|Ci)P(Ci)
    pP = sum(testWordsMarkedArray * pWordsPosicity * DS) + np.log(pPos)
    pN = sum(testWordsMarkedArray * pWordsNegy * DS_neg) + np.log(pNeg)
    pNeu = sum(testWordsMarkedArray * pWordsNeutral * DS_neutral) + np.log(pNeutral)

    if pP > pN > pNeu or pP > pNeu > pN:
        return pP, pN, pNeu, 1
    elif pN > pP > pNeu or pN > pNeu > pP:
        return pP, pN, pNeu, 2
    else:
        return pP, pN, pNeu, 3



def classify_two(pWordsSpamicity, pWordsHealthy, DS, pSpam, testWordsMarkedArray):
    ps = sum(testWordsMarkedArray * pWordsSpamicity * DS) + np.log(pSpam)
    ph = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pSpam)
    if ps > ph:
        return ps, ph, 1
    else:
        return ps, ph, 0