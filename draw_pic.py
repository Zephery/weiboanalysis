# SVM 准确率

from pylab import mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
fig, ax = plt.subplots()
x = [i for i in range(0, 21)]
array = [
    0.850000, 0.825000, 0.818750, 0.800000, 0.843750, 0.812500, 0.775000, 0.831250, 0.850000, 0.850000, 0.850000,
    0.806250, 0.875000, 0.850000, 0.831250, 0.850000, 0.875000, 0.856250, 0.837500, 0.862500, 0.842300
]
plt.plot(x, array, "x-", label="正确率")
plt.plot(x, [0.8375] * 21,"+-", label="平均正确率")
plt.xlabel("次数")
plt.ylabel("准确率")
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.title("SVC")
plt.show()












# 三类
# 1.画图，朴素贝叶斯，用自己实现的和sklearn中自带的MultinomialNB进行错误率比较
# from pylab import mpl
# import matplotlib.pyplot as plt
# import numpy as np
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# migtime = [0.37, 0.33, 0.32, 0.33, 0.35, 0.3, 0.32, 0.32, 0.35, 0.29]
# mid=[1-np.sum(migtime)/10]
# # delay = [0.44, 0.48, 0.38, 0.35, 0.36, 0.34, 0.35, 0.39, 0.40, 0.35]
# fig, ax = plt.subplots()
# plt.xlabel('次数')
# plt.ylabel('准确率')
# x = [i for i in range(1, 11)]
# plt.plot(x, [1-t for t in migtime], "x-", label="正确率")
# plt.plot(x,10*mid,label="平均正确率")
# plt.title("多项式朴素贝叶斯")
# # plt.plot(x, delay, "+-", label="实现代码")
# plt.grid(True)
# plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
# plt.show()





# 2.Adaboost的50次迭代过程错误率
# from pylab import mpl
# import matplotlib.pyplot as plt
# import numpy as np
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# x = [i for i in range(1, 51)]
# array = [0.270000, 0.200000, 0.170000, 0.190000, 0.170000, 0.160000, 0.190000, 0.160000, 0.180000, 0.180000, 0.170000,
#          0.160000, 0.180000, 0.170000, 0.180000, 0.170000, 0.170000, 0.160000, 0.190000, 0.170000, 0.170000, 0.170000,
#          0.170000, 0.170000, 0.190000, 0.160000, 0.170000, 0.170000, 0.180000, 0.170000, 0.180000, 0.160000, 0.170000,
#          0.180000, 0.180000, 0.160000, 0.180000, 0.160000, 0.180000, 0.180000, 0.170000, 0.160000, 0.180000, 0.170000,
#          0.180000, 0.170000, 0.170000, 0.160000, 0.190000, 0.170000]
# plt.plot(x, array,label="错误率")
# plt.xlabel("次数")
# plt.ylabel("错误率")
# plt.xlim(0.5,50)
# plt.title("AdaBoost二分类错误率")
# plt.show()
