import pynlpir
import re
pynlpir.open()


s = '不让我上桌吃饭我还不会自己抢吗！[doge][doge][doge]（投稿：@还没怀上的葛一他麻麻）http://t.cn/RqKTebK   '
stop = [line.strip() for line in open('ad/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
print(list(set(pynlpir.segment(s, pos_tagging=False))))
#['cn', '全民', 'R68kD0I', '饮酒', '醉', ' ', '一人', 'K', 't', '甜', '听听', '歌', '一首歌', '♥', 'http', '酸', '唱']
#['听听', '全民', '点', '@全民K歌', '首', ' ', '酸', 'http://t.cn/R68kD0I', '饮酒', '唱', '歌', '醉', 'K', '♥有点', '甜']

