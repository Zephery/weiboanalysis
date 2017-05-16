import random

from train.boostNB import *
from train.tool import *


def trainingAdaboostGetDS(iterateNum=50):
    MoodWordsArray, classLables = get_feature("train.txt")
    testWords = []
    testWordsType = []
    testCount = 100  # 从中随机选取100条用来测试，并删除原来的位置
    for i in range(testCount):
        try:
            randomIndex = int(random.uniform(0, len(MoodWordsArray)))
            testWordsType.append(classLables[randomIndex])
            testWords.append(MoodWordsArray[randomIndex])
            del (MoodWordsArray[randomIndex])
            del (classLables[randomIndex])
        except Exception as e:
            print(e)
    """
    训练阶段，可将选择的vocabularyList也放到整个循环中，以选出
    错误率最低的情况，获取最低错误率的vocabularyList
    """
    vocabularyList = build_key_word("train.txt")
    # vocabularyList=pynlp_build_key_word("train.txt")
    print("生成语料库！")
    trainMarkedWords = setOfWordsListToVecTor(vocabularyList, MoodWordsArray)
    print("数据标记完成！")
    trainMarkedWords = np.array(trainMarkedWords)
    print("数据转成矩阵！")
    PosWords, NegWords, NeutralWords, pPos, pNeg, pNeutral = \
        trainingNaiveBayes(trainMarkedWords, classLables)
    DS = np.ones(len(vocabularyList))
    DS_neg = np.ones(len(vocabularyList))
    DS_neutral = np.ones(len(vocabularyList))
    DS_temp = np.ones(len(vocabularyList))
    ds_errorRate = {}
    minErrorRate = np.inf
    temp = 0
    for i in range(iterateNum):
        errorCount = 0.0
        for j in range(testCount):
            try:
                testWordsArray = setOfWordsToVecTor(vocabularyList, testWords[j])
                # print(testWordsArray)
                if temp == 0:
                    pP, pN, pNeu, smsType = classify(PosWords, NegWords, NeutralWords,
                                                     DS, pPos, pNeg, pNeutral, testWordsArray, DS_neg, DS_neutral)
                elif temp == 1:
                    pP, pN, pNeu, smsType = classify(PosWords, NegWords, NeutralWords,
                                                     DS, pPos, pNeg, pNeutral, testWordsArray, DS_temp, DS_temp)
                elif temp == 2:
                    pP, pN, pNeu, smsType = classify(PosWords, NegWords, NeutralWords,
                                                     DS_temp, pPos, pNeg, pNeutral, testWordsArray, DS_neg, DS_temp)
                else:
                    pP, pN, pNeu, smsType = classify(PosWords, NegWords, NeutralWords,
                                                     DS_temp, pPos, pNeg, pNeutral, testWordsArray, DS_temp, DS_neutral)
                if smsType != testWordsType[j]:
                    # print(testWords[j],smsType,testWordsType[j])
                    errorCount += 1
                    if testWordsType[j] == 1:
                        alpNa = pP - pN - pNeutral
                        if alpNa < 0.5:
                            continue
                        DS[testWordsArray != 0] = np.abs(  # abs取绝对值
                            (DS[testWordsArray != 0] + np.exp(alpNa)) / DS[testWordsArray != 0])  # exp求指数
                        temp = 1
                    elif testWordsType[j] == 2:
                        alpNa = pN - pP - pNeutral
                        if alpNa < 0.5:
                            continue
                        DS_neg[testWordsArray != 0] = np.abs(  # abs取绝对值
                            (DS_neg[testWordsArray != 0] + np.exp(alpNa)) / DS_neg[testWordsArray != 0])
                        temp = 2
                    else:

                        alpNa = pNeutral - pP - pN
                        if alpNa < 0.5:
                            continue
                        DS_neutral[testWordsArray != 0] = np.abs(  # abs取绝对值
                            (DS_neutral[testWordsArray != 0] + np.exp(alpNa)) / DS_neutral[testWordsArray != 0])
                        temp = 3
                        # print(DS[testWordsArray])
            except Exception as e:
                traceback.print_exc(e)
        # print('DS:', DS)
        # print('DS_t:',DS_t)
        errorRate = errorCount / testCount
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            ds_errorRate['minErrorRate'] = minErrorRate
            ds_errorRate['DS'] = DS
        print('第 %d 轮迭代，错误个数 %d ，错误率 %f' % (i, errorCount, errorRate))
        if errorRate == 0.0:
            break
    # print(DS)
    # print(DS_neg)
    # print(DS_neutral)
    ds_errorRate['vocabularyList'] = vocabularyList
    ds_errorRate['PosWords'] = PosWords
    ds_errorRate['NegWords'] = NegWords
    ds_errorRate['NeutralWords'] = NeutralWords
    ds_errorRate['pPos'] = pPos
    return ds_errorRate


if __name__ == '__main__':
    dsErrorRate = trainingAdaboostGetDS()
    # 保存模型训练的信息
    # np.savetxt('PosWords.txt', dsErrorRate['PosWords'], delimiter='\t')
    # np.savetxt('NegWords.txt', dsErrorRate['NegWords'], delimiter='\t')
    # np.savetxt('pPos.txt', np.array([dsErrorRate['pPos']]), delimiter='\t')
    # np.savetxt('trainDS.txt', dsErrorRate['DS'], delimiter='\t')
    # np.savetxt('trainMinErrorRate.txt', np.array([dsErrorRate['minErrorRate']]), delimiter='\t')
    vocabulary = dsErrorRate['vocabularyList']
    posword = list(dsErrorRate['PosWords'])
    negword = list(dsErrorRate['NegWords'])
    neutralword = list(dsErrorRate['NeutralWords'])
    fw = open('vocabularyList.txt', 'w', encoding='utf-8')
    for i in range(len(vocabulary)):
        fw.write(vocabulary[i] + '\t' + str(posword[i]) + '\t' + str(negword[i]) + '\t' + str(neutralword[i]) + '\n')
    fw.flush()
    fw.close()
