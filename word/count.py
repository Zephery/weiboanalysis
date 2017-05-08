# coding:utf-8
import jieba
from .cut import CUT_METHODS



def word_count(sentence,method):
    """
    切割字符后计算词的出现频率
    :param sentence:
    :return:
    """
    res = {}
    word_list = CUT_METHODS.get(method)(sentence)
    for word in word_list:
        if len(word) > 1:
            res[word] = sentence.count(word)
    return res
