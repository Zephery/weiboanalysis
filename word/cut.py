# coding:utf-8
import jieba


def default_cut(sentence):
    """
    默认的切割方式
    :param sentence:
    :return:
    """
    return jieba.lcut(sentence=sentence)


def all_cut(sentence):
    """
    全模式的切割方式
    :param sentence:
    :return:
    """
    return jieba.lcut(sentence, cut_all=True)


def search_cut(sentence):
    """
    HMM的切割方式
    :param sentence:
    :return:
    """
    return jieba.lcut_for_search(sentence)


def search_all_cut(sentence):
    """
    HMM的切割方式
    :param sentence:
    :return:
    """
    return jieba.lcut_for_searchsen(sentence, cut_all=True)


CUT_METHODS = {
    "default": default_cut,
    "all": all_cut,
    "search": search_cut,
    "search_all": search_all_cut
}
