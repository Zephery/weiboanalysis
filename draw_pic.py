# 三类
# 1.画图，朴素贝叶斯，用自己实现的和sklearn中自带的MultinomialNB进行错误率比较
from pylab import mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']
migtime = [0.37, 0.33, 0.32, 0.33, 0.35, 0.3, 0.32, 0.32, 0.35, 0.29]
delay = [0.44, 0.48, 0.38, 0.35, 0.36, 0.34, 0.35, 0.39, 0.40, 0.35]
fig, ax = plt.subplots()
plt.xlabel('次数')
plt.ylabel('错误率')
x = [i for i in range(1, 11)]
plt.plot(x, migtime, "x-", label="MultinomialNB")
plt.plot(x, delay, "+-", label="实现代码")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.show()





# 2.Adaboost的50次迭代过程错误率
import matplotlib.pyplot as plt

x = [i for i in range(1, 51)]
array = [0.270000, 0.200000, 0.170000, 0.190000, 0.170000, 0.160000, 0.190000, 0.160000, 0.180000, 0.180000, 0.170000,
         0.160000, 0.180000, 0.170000, 0.180000, 0.170000, 0.170000, 0.160000, 0.190000, 0.170000, 0.170000, 0.170000,
         0.170000, 0.170000, 0.190000, 0.160000, 0.170000, 0.170000, 0.180000, 0.170000, 0.180000, 0.160000, 0.170000,
         0.180000, 0.180000, 0.160000, 0.180000, 0.160000, 0.180000, 0.180000, 0.170000, 0.160000, 0.180000, 0.170000,
         0.180000, 0.170000, 0.170000, 0.160000, 0.190000, 0.170000]
plt.plot(x, array)
plt.show()




