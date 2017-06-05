'''
使用机器学习库sklearn处理多分类问题
'''
import random
from itertools import cycle

import matplotlib.pyplot as plt
import random
import Bayes as bayes
import numpy as np

from pylab import mpl
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

from pylab import mpl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import label_binarize

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_estimators = 500
    learning_rate = 1.
    vocabList = bayes.build_key_word("../train/train.txt")
    line_cut, label = bayes.loadDataSet("../train/train.txt")
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
    multi = MultinomialNB()
    # multi = multi.fit(train_mood_array, label)     # 去掉效果更佳，否则为AdaBoost训练前就进行了训练
    ada_real = AdaBoostClassifier(
        base_estimator=multi,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME.R")
    ada_real.fit(train_mood_array, label)
    ada_real_err = np.zeros((n_estimators,))   #变成一个一维的矩阵，长度为n
    for i, y_pred in enumerate(ada_real.staged_predict(test_word_array)):  # 测试
        ada_real_err[i] = zero_one_loss(y_pred, test_word_arrayLabel) # 得出不同的，然后除于总数

    # ROC start
    # X_train, X_test, y_train, y_test = train_test_split(train_mood_array, label, test_size=.3,
    #                                                     random_state=0)
    # y = label
    # y = label_binarize(y, classes=[1, 2, 3])
    # n_classes = y.shape[1]
    # ada_real.fit(X_train, y_train)
    # y_score = ada_real.decision_function(X_test)
    # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # # plt.figure()
    # lw = 2
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'.format(i + 1, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    # ROC end

    # 画图、错误率
    ada_real_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(train_mood_array)):  # 训练样本对训练样本的结果
        ada_real_err_train[i] = zero_one_loss(y_pred, label)

    ax.plot(np.arange(n_estimators) + 1, ada_real_err,
            label='测试错误率',
            color='orange')
    ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
            label='训练错误率',
            color='green')
    ax.set_xlabel('次数')
    ax.set_ylabel('错误率')

    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.title("AdaBoost.SAMME.R")
    plt.show()
