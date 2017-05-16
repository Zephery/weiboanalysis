import random
import numpy as np
import traceback
from two_class import two_nb as two_nb


def trainingAdaboostGetDS(iterateNum=50):
    vocabList = two_nb.build_key_word("two_class.txt")
    line_cut, classLables = two_nb.loadDataSet("two_class.txt")
    trainMarkedWords = two_nb.setOfWordsListToVecTor(vocabList, line_cut)
    testWordsArray = []
    testWordsType = []
    testCount = 100  # 从中随机选取100条用来测试，并删除原来的位置
    for i in range(testCount):
        try:
            randomIndex = int(random.uniform(0, len(trainMarkedWords)))
            testWordsType.append(classLables[randomIndex])
            testWordsArray.append(trainMarkedWords[randomIndex])
            del (trainMarkedWords[randomIndex])
            del (classLables[randomIndex])
        except Exception as e:
            print(e)
    PosWords, NegWords, prior_Pos = two_nb.trainingNaiveBayes(trainMarkedWords, classLables)
    DS = np.ones(len(vocabList))
    DS_temp = np.ones(len(vocabList))
    ds_errorRate = {}
    minErrorRate = np.inf
    for i in range(iterateNum):
        errorCount = 0.0
        for j in range(testCount):
            try:
                ps, ph, smsType = two_nb.classify(PosWords, NegWords, prior_Pos, testWordsArray[j], DS, DS_temp)
                if smsType != testWordsType[j]:
                    # print(testWords[j],smsType,testWordsType[j])
                    errorCount += 1
                    alpNa = ps - ph
                    if alpNa > 0:
                        DS[testWordsArray[j] != 0] = np.abs(  # abs取绝对值
                            (DS[testWordsArray[j] != 0] - np.exp(alpNa)) / DS[testWordsArray[j] != 0])  # exp求指数
                    else:
                        DS[testWordsArray[j] != 0] = (DS[testWordsArray[j] != 0] + np.exp(alpNa)) / DS[
                            testWordsArray[j] != 0]
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
    ds_errorRate['vocabularyList'] = vocabList
    ds_errorRate['PosWords'] = PosWords
    ds_errorRate['NegWords'] = NegWords
    return ds_errorRate


if __name__ == '__main__':
    dsErrorRate = trainingAdaboostGetDS()
    # 保存模型训练的信息
    # np.savetxt('PosWords.txt', dsErrorRate['PosWords'], delimiter='\t')
    # np.savetxt('NegWords.txt', dsErrorRate['NegWords'], delimiter='\t')
    # np.savetxt('pPos.txt', np.array([dsErrorRate['pPos']]), delimiter='\t')
    # np.savetxt('trainDS.txt', dsErrorRate['DS'], delimiter='\t')
    # np.savetxt('trainMinErrorRate.txt', np.array([dsErrorRate['minErrorRate']]), delimiter='\t')
    # vocabulary = dsErrorRate['vocabularyList']
    # posword = list(dsErrorRate['PosWords'])
    # negword = list(dsErrorRate['NegWords'])
    # neutralword = list(dsErrorRate['NeutralWords'])
    # fw = open('vocabularyList.txt', 'w', encoding='utf-8')
    # for i in range(len(vocabulary)):
    #     fw.write(vocabulary[i] + '\t' + str(posword[i]) + '\t' + str(negword[i]) + '\t' + str(neutralword[i]) + '\n')
    # fw.flush()
    # fw.close()
