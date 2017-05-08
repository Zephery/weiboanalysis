import pynlpir
import re
pynlpir.open()


s = '夭寿咯'
stop = [line.strip() for line in open('ad/stop.txt', 'r', encoding='utf-8').readlines()]  # 停用词
print(list(set(pynlpir.segment(s, pos_tagging=False))))
#['cn', '全民', 'R68kD0I', '饮酒', '醉', ' ', '一人', 'K', 't', '甜', '听听', '歌', '一首歌', '♥', 'http', '酸', '唱']
#['听听', '全民', '点', '@全民K歌', '首', ' ', '酸', 'http://t.cn/R68kD0I', '饮酒', '唱', '歌', '醉', 'K', '♥有点', '甜']

