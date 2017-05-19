import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from two_class import two_nb as two_nb


def find_error(DS, test_word_array, test_word_arrayLabel, testCount, PosWords, NegWords, prior_Pos, DS_temp):
    result_type = []
    errorCount = 0.0
    for j in range(testCount):
        ps, ph, smsType = two_nb.classify(PosWords, NegWords, prior_Pos, test_word_array[j], DS, DS_temp)
        result_type.append(smsType)
        if smsType != test_word_arrayLabel[j]:
            errorCount += 1
    return float(errorCount) / testCount, result_type


def trainingAdaboostGetDS(iterateNum=50):
    vocabList = two_nb.build_key_word("two_class.txt")
    line_cut, label = two_nb.loadDataSet("two_class.txt")
    train_mood_array = two_nb.setOfWordsListToVecTor(vocabList, line_cut)
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
    DS = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    DS_temp = np.mat(np.ones((m, 1)) / m)  # 引入一个临时的DS_temp，用来保持一致
    PosWords, NegWords, prior_Pos = two_nb.trainingNaiveBayes(train_mood_array, label)
    ds_errorRate = {}
    error_list=[]
    minErrorRate = np.inf
    ROC_value=[]
    for i in range(iterateNum):  # 进行50次迭代
        errorRate, result_type = find_error(DS, test_word_array, test_word_arrayLabel, testCount, PosWords,
                                            NegWords, prior_Pos, DS_temp)
        if errorRate > 0.4:  # 设置阈值
            break
        alpha = float(
            0.5 * np.log((1.0 - errorRate) / errorRate))
        Z = np.sum(DS * np.exp(-alpha))
        for j in range(len(result_type)):
            if result_type[j] != test_word_arrayLabel[j]:
                DS[test_word_array[j] != 0] = DS[test_word_array[j] != 0] * np.exp(alpha) / Z
            else:
                DS[test_word_array[j] != 0] = DS[test_word_array[j] != 0] * np.exp(-alpha) / Z
            # ROC
        TPR, FPR=print_metrics(test_word_arrayLabel,result_type)
        roc_tuple=(TPR,FPR)
        ROC_value.append(roc_tuple)
        error_list.append(errorRate)
        if errorRate < minErrorRate:
            minErrorRate = errorRate
            ds_errorRate['minErrorRate'] = minErrorRate
            ds_errorRate['DS'] = DS
        print('第 %d 轮迭代，错误率 %f' % (i, errorRate))
        if errorRate == 0.0:
            break
    ds_errorRate['ROC_value']=ROC_value
    ds_errorRate['errorRate']=error_list
    return ds_errorRate


def print_metrics(test_word_arrayLabel, result_type):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    num_examples = len(test_word_arrayLabel)
    for example_num in range(0, num_examples):
        predicted_label = result_type[example_num]
        if test_word_arrayLabel[example_num] == 1:
            if predicted_label == 1:
                true_positives += 1
            elif predicted_label == 2:
                false_negatives += 1
        elif test_word_arrayLabel[example_num] == 2:
            if predicted_label == 1:
                false_positives += 1
            elif predicted_label == 2:
                true_negatives += 1
    TPR=true_positives/(true_positives+false_negatives)
    FPR=false_positives/(true_negatives+false_positives)
    return TPR,FPR


# def plotROCCurve(ROC_value):
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(2):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


def print_error_Rate(error_Rate):
    x=[t for t in range(len(error_Rate))]
    error_Rate=[t for t in error_Rate]
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    ax.plot(x,error_Rate)
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve for AdaBoost horse colic detection system')
    # ax.axis([0, 50])
    plt.show()

if __name__ == '__main__':
    dsErrorRate = trainingAdaboostGetDS()
    try:
        print_error_Rate(dsErrorRate['errorRate'])
        # plotROCCurve(dsErrorRate['ROC_value'])
        np.savetxt('DS.txt', np.array([dsErrorRate['DS']]), delimiter='\n')
    except Exception as e:
        print(e)
        print("错误率大于阈值，重试")
