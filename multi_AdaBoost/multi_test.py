import random
import Bayes as bayes
import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

if __name__=='__main__':
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
    multi=MultinomialNB()
    multi=multi.fit(train_mood_array,label)
    ada_real = AdaBoostClassifier(
        base_estimator=multi,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME.R")
    ada_real.fit(train_mood_array,label)

    ada_real_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(test_word_array)):
        ada_real_err[i] = zero_one_loss(y_pred, test_word_arrayLabel)


    ada_real_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(train_mood_array)):
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