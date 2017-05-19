import random
import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from two_class import two_nb as two_nb

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    n_estimators = 200
    learning_rate = 1.
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
    multi = MultinomialNB()
    multi.fit(train_mood_array, label)
    multi.predict(test_word_array)
    ada_real = AdaBoostClassifier(
        base_estimator=multi,
        learning_rate=learning_rate,
        n_estimators=n_estimators)
    ada_real.fit(train_mood_array, label)
    ada_real_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(test_word_array)):
        ada_real_err[i] = zero_one_loss(y_pred, test_word_arrayLabel)
        print(ada_real_err[i])

    # ROC    start
    X_train = train_mood_array
    X_test = test_word_array
    y_train = label
    y_test = test_word_arrayLabel
    y = label
    y_score = ada_real.predict_proba(test_word_array)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in [0, 1]:
        fpr[i], tpr[i], _ = roc_curve(test_word_arrayLabel, y_score[:, 0], pos_label=1)
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('二分类AdaBoost的ROC曲线')
    plt.legend(loc="lower right")
    plt.show()
    # ROC end




    # 错误率
    # ax.plot(np.arange(n_estimators) + 1, ada_real_err,
    #         label='测试错误率',
    #         color='orange')
    # ax.set_xlabel('次数')
    # ax.set_ylabel('错误率')
    # leg = ax.legend(loc='upper right', fancybox=True)
    # leg.get_frame().set_alpha(0.7)
    # plt.title("二元分类")
    # plt.show()
