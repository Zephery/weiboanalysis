# 基于AdaBoost算法的情感分析研究  
**此项目为本科毕业设计项目，目前已经没有时间更新了，文章、代码都有很多错误，大家借鉴一下思路就好，不要仔细研究** 

**大学时没有好好学算法，毕竟那些树、图实在提不起兴趣，好在毕业设计选择了个机器学习算法，整了个还算是有点意思的项目，至少弥补了大学的一点点的遗憾。现在将项目开源出来，虽然感觉还是写得没有达到自己的预期，大部分也是参考别人的，有兴趣的可以下载看看吧。如果可以，希望能给个star或者fork奖励奖励**

**文本分类基本流程**
<div align="center">

![](http://image.wenzhihuai.com/images/20171217043631.png)

</div>

## 运行环境  
[anaconda: 3.5+]https://www.anaconda.com/

## 本文项目流程
一、 使用微博应用获取微博文本，代码地址[weibo_get](https://github.com/Zephery/weibo_get)  
二、 SVM初步分类(svm_temp.py)  
三、 利用贝叶斯定理进行情感分析  
四、 利用AdaBoost加强分类器
  
**完整文档可以看doc**
[https://github.com/Zephery/weiboanalysis/blob/master/doc](https://github.com/Zephery/weiboanalysis/blob/master/doc)

## 一、获取微博文本
<div align="center">

![](http://image.wenzhihuai.com/images/20171217053231.png)

</div>



## 二、SVM初步分类
<div align="center">

![](http://image.wenzhihuai.com/images/20171217053051.png)

</div>



## 三、使用朴素贝叶斯分类
<div align="center">

![](http://image.wenzhihuai.com/images/20171217043913.png)

</div>


## 四、AdaBoost
#### 4.1 二分类AdaBoost
<div align="center">

![](http://image.wenzhihuai.com/images/20171217043935.png)

</div>

#### 4.2 多分类AdaBoost
**4.2.1 AdaBoost.SAMME**
<div align="center">

![](http://image.wenzhihuai.com/images/20171217043944.png)

</div>

**4.2.2 AdaBoost.SAMME.R**
<div align="center">

![](http://image.wenzhihuai.com/images/20171217043951.png)

</div>

