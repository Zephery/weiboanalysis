# encoding=utf-8
import jieba
import re
jieba.load_userdict("train/word.txt")
line='2 午睡还被说怎么还能够睡得心安理得。。。'
# s=line
# p = re.compile(r'http?://.+$')  # 正则表达式，提取URL
# result = p.findall(line)  # 找出所有url
# if len(result):
#     for i in result:
#         s = s.replace(i, '')  # 一个一个的删除
# seg_list = jieba.cut(str（s).replace(' ', ''))  # 默认是精确模式
stop = [line.strip() for line in open('ad/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
print(list(set(jieba.cut(line)) - set(stop)))
print(list(jieba.cut(line)))