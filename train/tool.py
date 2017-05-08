import re
import pynlpir
import jieba
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB

jieba.load_userdict("word.txt")
pynlpir.open()
stop = [line.strip() for line in open('../ad/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词

def load_key_words(file_path):
    with open(file_path, encoding="utf-8") as fp:
        lines = fp.readlines()
        lines = [line.replace("\n", "") for line in lines]
    return lines


def pynlp_build_key_word(filename):
    d={}
    with open(filename, encoding="utf-8") as fp:
        for line in fp:
            s = line
            p = re.compile(r'http?://.+$')  # 正则表达式，提取URL
            result = p.findall(line)  # 找出所有url
            if len(result):
                for i in result:
                    s = s.replace(i, '')  # 一个一个的删除
            temp = pynlpir.segment(s, pos_tagging=False)  # 分词
            for i in temp:
                if '@' in i:
                    temp.remove(i)  # 删除分词中的名字
                p = re.compile(r'\w', re.L)
                result = p.sub("", i)
                if not result or result == ' ':  # 空字符
                    continue
                if len(i) > 1:  # 避免大量无意义的词语进入统计范围
                    d[i] = d.get(i, 0) + 1
    kw_list = sorted(d, key=lambda x: d[x], reverse=True)
    size = int(len(kw_list) * 0.2)  # 取最前的30%
    mood = set(kw_list[:size])
    return list(mood - set(stop)- set('\u200b') - set(' ') - set('\u3000'))

def build_key_word(path):  # 通过词频产生key word
    d = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            for word in jieba.cut(line.strip()):
                p = re.compile(r'\w', re.L)
                result = p.sub("", word)
                if not result or result == ' ':  # 空字符
                    continue
                if len(word) > 1:  # 避免大量无意义的词语进入统计范围
                    d[word] = d.get(word, 0) + 1
    kw_list = sorted(d, key=lambda x: d[x], reverse=True)
    size = int(len(kw_list) * 0.15)  # 取最前的30%
    mood = set(kw_list[:size])
    return list(mood - set(stop))


def getlinejieba(path):
    d = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            temp = []
            line = str(line).replace('\u200b', '')
            for word in jieba.cut(line.strip()[2:]):
                temp.append(word)
            d.append(list(set(temp) - set(stop) - set(' ')))  # 差集、去空
    return d


def _get_feature(sentence, key_word):
    size = len(key_word)
    feature = [0 for _ in range(size)]  # 初始化矩阵，全为0
    for index in range(size):
        word = key_word[index]
        value = sentence.find(word)  # 单词最初出现的位置
        if value != -1:
            feature[index] += 1  # 进行累加
    return np.array(feature)


def get_word_feature(sentence):
    wordlist = []
    sentence = str(sentence).replace('\u200b', '')
    for word in jieba.cut(sentence.strip()):
        p = re.compile(r'\w', re.L)
        result = p.sub("", word)
        if not result or result == ' ':  # 空字符
            continue
        wordlist.append(word)
    return list(set(wordlist) - set(stop) - set(' '))


def get_feature(path):
    features = []
    label = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            temp = line.strip()
            try:
                sentence = temp[2:].lstrip()  # 每条微博
                label.append(int(temp[:2]))  # 获取标注
                # features.append(_get_feature(sentence, kw_list))
                features.append(get_word_feature(sentence))
            except Exception:
                # print(temp + " error")
                continue
    return features, label  # 返回每条微博的特征的矢量化和标注


def script_run():
    # 产生keyword
    kw_list = build_key_word("train.txt")
    # 保存数据
    fp = open("new_word.txt", encoding="utf-8", mode="w")
    for word in kw_list:
        fp.write(word + "\n")
    fp.close()
    # kw_list = load_key_words("word.txt")
    feature, label = get_feature("train.txt", kw_list)
    gnb = MultinomialNB()  # 多项式贝叶斯
    gnb = gnb.fit(feature, label)
    joblib.dump(gnb, 'model/gnb.model')
    print("训练完成")


def test(test_data, model_name):
    kw_list = load_key_words("new_word.txt")
    feature_list = []
    for data in test_data:
        feature_list.append(_get_feature(data, kw_list))
    gnb = joblib.load(model_name)
    result = gnb.predict(feature_list)
    for i in range(len(test_data)):
        print(test_data[i], "----", result[i])


if __name__ == "__main__":
    # script_run()
    # test(["不惊扰别人的宁静，就是慈悲； 不伤害别人的自尊，就是善良。 人活着，发自己的光就好，不要吹灭别人的灯。",
    #       "再次梦见老同学老朋友，这几年多少次梦见过他我已数不清，对于我们之间，在我人生的某一阶级是很好的记忆，缘份这东西很奇妙，形容我们之间只能用有缘无份，都有互相的联系方式，却都从未拨通过那个号码，点击过那个图像。遗憾都在我们心底…… ​",
    #       "与大哥太有缘分！竟是同天生日！！愿大哥的模特演绎之路越走越远越走越顺越走越好！！！",
    #       "萌萌哒",
    #       "现在的年轻人，连点小事都做不好",
    #       "丁丁美人飞机临时取消，只能明天再见了[失望]",
    #       "默默看书"], "model/gnb.model")
    for i in range(0,12):
        vo=build_key_word("train.txt")
        print('')