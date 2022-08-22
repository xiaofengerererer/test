# coding=utf-8
import os

import matplotlib.pyplot as plt
import pandas as pd
# 计算每个类别的数量
sample_dir = '../VOCdevkit/VOC2007/YOLOLabels'  # 标签文件所在的路径
filenames = os.listdir(sample_dir)

class_list = []
anno_num = 0

#遍历文件获得类别列表
for filename in filenames:
    if '.txt' in filename:
        label_file = sample_dir + '/' + filename
        with open(label_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                curLine = line.strip().split(" ")
                label = curLine[0]   # 以DOTA格式的标签为例，获取label字段（其他格式的按读取方式获取字段即可）
                if label not in class_list:
                    class_list.append(label)

class_num = len(class_list)
EachClass_Num = {}
for i in range(class_num):
    EachClass_Num[class_list[i]] = 0

for filename in filenames:
    if '.txt' in filename:
        label_file = sample_dir + '/' + filename
        with open(label_file, 'r', encoding='gbk') as f:
            for line in f.readlines():
                curLine = line.strip().split(" ")
                label_list = curLine[0]
                label = ''.join([str(x) for x in label_list])
                if label:
                    EachClass_Num[label] = EachClass_Num[label] + 1  # 统计各类别的目标个数
                else:
                    continue

print(EachClass_Num)
# x = EachClass_Num[0]
# print(x)
# plt.hist()

## 保存输出
data_out = []
for key in EachClass_Num:
    k = [key, EachClass_Num[key]]
    data_out.append(k)


# list转dataframe
df = pd.DataFrame(data_out, columns=['class', 'num'])

# 保存到本地excel
df.to_excel("./val_dota_class.xlsx", index=False, encoding='gbk')
