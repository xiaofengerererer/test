#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 06:42:30 2018
@author: pc
"""

import os
from pyecharts.charts import Bar
import os.path
import math
import xml.etree.cElementTree as et
from scipy.ndimage import measurements
#
# 画面积大小直方图有
path= r"C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\Annotations"
files=os.listdir(path)
s=[]
ratio_list = []

def file_extension(path):
    return os.path.splitext(path)[1]

for xmlFile in files:
    if not os.path.isdir(xmlFile):
        if file_extension(xmlFile) == '.xml':
            tree=et.parse(os.path.join(path,xmlFile))
            root=tree.getroot()
            filename=root.find('filename').text
            square1_list = []
            side1_list = []
            # ratio_list = []
            for Object in root.findall('size'):  #the size of pic
                side1=Object.find('width').text
                side2=Object.find('height').text
                area = float(side1) *float(side2)

            for Object in root.findall('object'):
                bndbox=Object.find('bndbox')
                xmin=bndbox.find('xmin').text
                ymin=bndbox.find('ymin').text
                xmax=bndbox.find('xmax').text
                ymax=bndbox.find('ymax').text
                square = (int(ymax)-int(ymin)) * (int(xmax)-int(xmin))
                square1_list.append(square)
#                print(xmin,ymin,xmax,ymax)
#                 print(square)
            for target in square1_list:
                # ratio = target /area
                ratio = math.sqrt(target / area) *100
                ratio_list.append(ratio)
max = max(ratio_list)
min = min(ratio_list)
num = 30 #最大面积
histogram1 = measurements.histogram(ratio_list,0,num,10) #直方图
histogram1 = list(map(int, histogram1)) #转换成 int 格式
print("histogram is ", histogram1)
bar = Bar()
bar.add_xaxis([str(num/10), str(num/10*2), str(num/10*3), str(num/10*4), str(num/10*5), str(num/10*6), str(num/10*7), str(num/10*8), str(num/10*9), str(num/10*10)])

bar.add_yaxis("数目", histogram1 )
bar.render(r"chart.html")#保存路径

print("图像搞定")
