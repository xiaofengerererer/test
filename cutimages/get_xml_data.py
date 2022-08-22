#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import xml.sax
import cv2

# TODO: xml文件路径、要保存的txt文件路径， 获取xml路径下所有文件名
xml_path = r"C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\Annotations"
txt_path = r"C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\YOLOLabels1"
images_path = r"C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\JPEGImages"
file_names = os.listdir(xml_path)

class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
# 初始宽高
img_w = 2000
img_h = 1500
list_row = []
list_col = []
list_child = []


class MovieHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.filename = ""
        self.name = ""
        self.xmin = 0
        self.ymin = 0
        self.xmax = 0
        self.ymax = 0

        self.class_num = -1
        self.content = ""
        self.flag = 1
    # 元素开始事件处理，遇到xml标签时候就调用
    def startElement(self, tag, attributes):
        self.CurrentData = tag

    # 元素结束事件处理，遇到xml的结束标签就调用
    def endElement(self, tag):
        if tag == "object" and self.flag == 1:
            # print("*****object*****")
            self.content += "\n"
        self.CurrentData = ""

    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "name":
            self.name = content
            self.class_num = class_names.index(content)
        elif self.CurrentData == "filename":
            self.filename = content
        elif self.CurrentData == "xmin":
            self.xmin = int(content)
        elif self.CurrentData == "ymin":
            self.ymin = int(content)
        elif self.CurrentData == "xmax":
            self.xmax = int(content)
        elif self.CurrentData == "ymax":
            self.ymax = int(content)
            # class center_x center_y width heigh
            # object_str = str(self.class_num) + " " + \
            #              str((self.xmin+self.xmax)/2/w) + " " + str((self.ymin+self.ymax)/2/h) + " "+\
            #              str((self.xmax-self.xmin)/w) + " " + str((self.ymax-self.ymin)/h)
            # self.content += object_str

            # 0原图 1行融合图 2列融合图
            # 优先判断是否在小图
            name = ""
            if where_main(self.xmin, self.xmax, self.ymin, self.ymax) != -1:
                self.flag = 1
                mix_num = str(where_main(self.xmin, self.xmax, self.ymin, self.ymax))
                name = "0 " + mix_num + " " + str(self.filename.split('.')[0]) + "_" + mix_num
            elif is_row_mix(self.xmin, self.xmax, self.ymin, self.ymax) != -1:
                flag = 1
                mix_num = str(is_row_mix(self.xmin, self.xmax, self.ymin, self.ymax))
                name = "1 " + mix_num + " " + str(self.filename.split('.')[0]) + "_mix_row_" + mix_num

            elif is_col_mix(self.xmin, self.xmax, self.ymin, self.ymax) != -1:
                self.flag = 1
                mix_num = str(is_col_mix(self.xmin, self.xmax, self.ymin, self.ymax))
                name = "2 " + mix_num + " " + str(self.filename.split('.')[0]) + "_mix_col_" + mix_num
            else:
                self.flag = 0
                print(f"error, {self.filename}, {self.xmin}, {self.xmax}, {self.ymin}, {self.ymax}")

            if self.flag == 1:
                object_str = name + " " + str(img_w) + " " + str(img_h) + " " + str(self.class_num) + " " + str(
                    self.xmin) + " " + str(self.xmax) + " " + str(self.ymin) + " " + str(self.ymax)
                self.content += object_str


def init_mix(w, h, split, mix_percent):
    list_row.clear()
    list_col.clear()
    list_child.clear()
    row_height = h // split
    col_width = w // split
    for i in range(4):
        mix_height_start = i * row_height
        mix_height_end = (i + 1) * row_height
        for j in range(3):
            mix_row_start = int(j * col_width + col_width * (1 - mix_percent))
            mix_row_end = int(mix_row_start + col_width * mix_percent * 2)
            list_row.append((mix_row_start, mix_row_end, mix_height_start, mix_height_end))
    # 保存成 xmin xmax ymin ymax
    for i in range(3):
        mix_col_start = int(i * row_height + row_height * (1 - mix_percent))
        mix_col_end = int(mix_col_start + row_height * mix_percent * 2)
        for j in range(4):
            mix_width_start = j * col_width
            mix_width_end = (j + 1) * col_width
            list_col.append((mix_width_start, mix_width_end, mix_col_start, mix_col_end))
    for i in range(4):
        height_start = i * row_height
        height_end = (i + 1) * row_height
        for j in range(4):
            width_start = j * col_width
            width_end = (j + 1) * col_width
            list_child.append((width_start, width_end, height_start, height_end))
    # print(list_child)


def is_row_mix(xmin, xmax, ymin, ymax):
    # 先判断是否在 行 混合区中
    i = 0
    for mix_row in list_row:
        if xmin >= mix_row[0] and xmax <= mix_row[1] and ymin >= mix_row[2] and ymax <= mix_row[3]:
            # print("row", xmin, xmax, ymin, ymax, mix_row)
            return i
        i += 1
    return -1


def is_col_mix(xmin, xmax, ymin, ymax):
    # 先判断是否在 列 混合区中
    i = 0
    for mix_col in list_col:
        if xmin >= mix_col[0] and xmax <= mix_col[1] and ymin >= mix_col[2] and ymax <= mix_col[3]:
            # print("col", xmin, xmax, ymin, ymax, mix_col)
            return i
        i += 1
    return -1


def where_main(xmin, xmax, ymin, ymax):
    # 判断在哪一个小图中
    i = 0
    for child in list_child:
        if xmin >= child[0] and xmax <= child[1] and ymin >= child[2] and ymax <= child[3]:
            return i
        i += 1
    return -1


# if __name__ == "__main__":
    # 创建一个 XMLReader
def main():
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    for file_name in file_names:
        # print(file_name)
        # 读取图片信息，此处命名规则是 xxxx.xml
        img = cv2.imread(images_path + "\\" + file_name.split('.')[0] + ".jpg")
        # 更改宽高
        sp = img.shape[0:2]
        img_h = sp[0]
        img_w = sp[1]
        init_mix(img_w, img_h, 4, 0.2)

        # 重写 ContextHandler
        Handler = MovieHandler()
        parser.setContentHandler(Handler)
        parser.parse(xml_path + "/" + file_name)
        file_name = file_name[:len(file_name) - 4]
        with open(txt_path + "/" + file_name + ".txt", "w") as f:
            f.write(Handler.content)
        f.close()
    print('finished the get_xml_data.py')
