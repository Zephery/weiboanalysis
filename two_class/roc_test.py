import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

##设置y值：表示实际值
y = np.array([1, 1, 2, 2])
##设置pred值：表示预测后的值
pred = np.array([0.1, 0.4, 0.35, 0.8])
##计算相关数据：注意返回的结果顺序
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
##计算曲线下面积
roc_auc=metrics.auc(fpr, tpr)
##绘图
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()