import random
import numpy as np
from multi_AdaBoost import Bayes as bayes
def find_error(DS, test_word_array, test_word_arrayLabel, testCount, PosWords, NegWords, prior_Pos,DS_temp):
    result_type = []
    errorCount = 0.0
    for j in range(testCount):
        ps, ph, smsType = bayes.classify(PosWords, NegWords, prior_Pos, test_word_array[j], DS,DS_temp)
        result_type.append(smsType)
        if smsType != test_word_arrayLabel[j]:
            errorCount += 1
    return float(errorCount) / testCount, result_type


def trainingAdaboostGetDS(iterateNum=50):
    vocabList = bayes.build_key_word("two_class.txt")
    line_cut, label = bayes.loadDataSet("two_class.txt")
    train_mood_array = bayes.setOfWordsListToVecTor(vocabList, line_cut)
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
    m = len(vocabList)
    DS = np.mat(np.ones((m, 1)) / m)
    DS_temp = np.mat(np.ones((m, 1)) / m)
    PosWords, NegWords, prior_Pos = bayes.trainingNaiveBayes(train_mood_array, label)
    ds_errorRate = {}
    minErrorRate = np.inf
    for i in range(iterateNum):
        errorRate, result_type = find_error(DS, test_word_array, test_word_arrayLabel, testCount, PosWords,
                                            NegWords,prior_Pos,DS_temp)
        if errorRate > 0.5:
            break
        alpha = float(
            0.5 * np.log((1.0 - errorRate) / errorRate))
        Z = np.sum(DS * np.exp(-alpha))
        for j in range(len(result_type)):
            if result_type[j] != test_word_arrayLabel[j]:
                DS[test_word_array[j]!=0] = DS[test_word_array[j]!=0] * np.exp(alpha) / Z
            else:
                DS[test_word_array[j]!=0] = DS[test_word_array[j]!=0] * np.exp(-alpha) / Z
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            ds_errorRate['minErrorRate'] = minErrorRate
            ds_errorRate['DS'] = DS
        print('第 %d 轮迭代，错误率 %f' % (i, errorRate))
        if errorRate == 0.0:
            break
    return ds_errorRate


if __name__ == '__main__':
    dsErrorRate = trainingAdaboostGetDS()
    np.savetxt('DS.txt', np.array([dsErrorRate['DS']]), delimiter='\n')

