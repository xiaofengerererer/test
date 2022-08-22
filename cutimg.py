'''实现分辨率为w*h裁剪成多张a*a大小的图片（a<=w且a<=h），以大小为a/2的步长滑动，裁剪同时修改标注框。'''
 
# coding:utf-8
import cv2
import os
import codecs
 
import xml.dom.minidom as xmldom
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
 
sys.setrecursionlimit(10000)
 
 
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint16)
def voc_xml_parse(xml_path):
    object_list = []
    domobj = xmldom.parse(xml_path)
    elementobj = domobj.documentElement
    folderobj = elementobj.getElementsByTagName("folder")[0]
    filenameobj = elementobj.getElementsByTagName("filename")[0]
    # sourceobj = elementobj.getElementsByTagName("source")[0]
    # ownerobj = elementobj.getElementsByTagName("owner")[0]
    sizeobj = elementobj.getElementsByTagName("size")[0]
    # segmentedobj = elementobj.getElementsByTagName("segmented")[0]
    head = {'folder': folderobj, 'filename': filenameobj, 'size': sizeobj,}
    object_list = elementobj.getElementsByTagName("object")
    return head, object_list
 
 
def voc_xml_modify(modify_xml_path, head, object_list):
    dom = xmldom.Document()
    root = dom.createElement('annotation')
    dom.appendChild(root)
    for obj in head.values():
        root.appendChild(obj)
    for obj in object_list:
        root.appendChild((obj))
    with open(modify_xml_path, 'w', encoding='utf-8') as f:
        dom.writexml(f, addindent='\t', newl='\n', encoding='utf-8')
    return
 
 
def crop_xml_modify(head, objectlist, hmin, wmin, new_height, new_width, origin_xml__path):
    filenameobj = head['filename']
    sizeobj = head['size']
    width = sizeobj.getElementsByTagName('width')[0]
    width.childNodes[0].data = str(new_width)
    # print(str(WIDTH))
    height = sizeobj.getElementsByTagName('height')[0]
    height.childNodes[0].data = str(new_height)
    # tree = ET.parse(origin_xml__path)
    # root = tree.getroot()
    obj = objectlist
    i = 0
    while (i < obj.length):
        # for obj in objectlist1:
        bndbox = obj[i].getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        XMIN = float(xmin.childNodes[0].data)
        ymin = bndbox.getElementsByTagName('ymin')[0]
        YMIN = float(ymin.childNodes[0].data)
        xmax = bndbox.getElementsByTagName('xmax')[0]
        XMAX = float(xmax.childNodes[0].data)
        ymax = bndbox.getElementsByTagName('ymax')[0]
        YMAX = float(ymax.childNodes[0].data)
        if (XMIN >= wmin) and (XMAX <= (wmin + new_width)) and (YMIN >= hmin) and (YMAX <= (hmin + new_height)):
            xmin.childNodes[0].data = str(int(XMIN - wmin))
            xmax.childNodes[0].data = str(int(XMAX - wmin))
            ymin.childNodes[0].data = str(int(YMIN - hmin))
            ymax.childNodes[0].data = str(int(YMAX - hmin))
        else:
            obj.remove(obj[i])
            i = i - 1  # 一定要向前提一个位置 删除的话用for是会出错的 耽搁了好久。。。
            # obj = objectlist1[i-1]
        i = i + 1
    return head, obj
 
 
def crop_dataset(imgpath, output_shape, annotation, cropAnno, cropImg, stride):
    origin_image = cv2.imread(imgpath)
    # image = Image.open(imgpath)
    # image_np = load_image_into_numpy_array(image)
    # origin_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    height, width = origin_image.shape[:2]
    # print(height)
    # print(width)
    domobj = xmldom.parse(annotation)
    elementobj = domobj.documentElement
    name = elementobj.getElementsByTagName("name")
    size = len(name)
    # tree = ET.parse(origin_xml__path)
    # root = tree.getroot()
    x = 0
    newheight = output_shape
    newwidth = output_shape
    while x < width:
        y = 0
        if x + newwidth <= width:
            while y < height:
                # 裁剪为output_shape*output_shape
                # newheight = output_shape
                # newwidth = output_shape
                head, objectlist = voc_xml_parse(annotation)
                if y + newheight <= height:
                    hmin = y
                    hmax = y + newheight
                    wmin = x
                    wmax = x + newwidth
                else:
                    hmin = height - newheight
                    hmax = height
                    wmin = x
                    wmax = x + newwidth
                    y = height  # test
                modify_head, modify_objectlist = crop_xml_modify(head, objectlist, hmin, wmin, newheight, newwidth,
                                                                 origin_xml__path)
                cropAnno1 = cropAnno + '_' + str(wmax) + '_' + str(hmax) + '_' + str(output_shape) + '.xml'
                cropImg1 = cropImg + '_' + str(wmax) + '_' + str(hmax) + '_' + str(output_shape) + '.jpg'
 
 
 
                import random
                throud = 3
 
 
                if len(modify_objectlist)>0 or random.randint(0, 100) < throud:
                    cv2.imwrite(cropImg1, origin_image[hmin: hmax, wmin: wmax])
                    voc_xml_modify(cropAnno1, modify_head, modify_objectlist)
 
                y = y + stride
                if y + output_shape == height:  # 第一张图就已经涵盖了height*height
                    y = height
                # if y + newheight > height:
                #     break
        else:
            while y < height:
                # 裁剪为output_shape*output_shape
                # newheight = output_shape
                # newwidth = output_shape
                head, objectlist = voc_xml_parse(annotation)
                if y + newheight <= height:
                    hmin = y
                    hmax = y + newheight
                    wmin = width - newwidth
                    wmax = width
                else:
                    hmin = height - newheight
                    hmax = height
                    wmin = width - newwidth
                    wmax = width
                    y = height  # test
                modify_head, modify_objectlist = crop_xml_modify(head, objectlist, hmin, wmin, newheight, newwidth,
                                                                 origin_xml__path)
                cropAnno1 = cropAnno + '_' + str(wmax) + '_' + str(hmax) + '_' + str(output_shape) + '.xml'
                cropImg1 = cropImg + '_' + str(wmax) + '_' + str(hmax) + '_' + str(output_shape) + '.jpg'
                if  len(modify_objectlist)> 0 or random.randint(0, 100) < throud:
                    cv2.imwrite(cropImg1, origin_image[hmin: hmax, wmin: wmax])
                    voc_xml_modify(cropAnno1, modify_head, modify_objectlist)
                y = y + stride
                # if y + newheight > height:
                #     break
            x = width
        x = x + stride
        if x + output_shape == width:  #第一张图就已经涵盖了height*height
            x = width
        # if x + newwidth > width:
        #     break
 
 
if __name__ == '__main__':
    # output_shape 为想要裁剪成的图片尺寸
    output_shape = 640  # 512 1024
    stride = int(output_shape / 4 * 3)  #重合25%
    imgpath = './VOCdevkit/VOC2007/JPEGImages='  # 原图路径
    annotation = './VOCdevkit/VOC2007/Annotations='  # 原图对应的标注xml文件路径
    cropAnno = './VOCdevkit/VOC2007/Annotations'  # 裁剪后存储xml的路径
    cropImg = './VOCdevkit/VOC2007/JPEGImages'  # 裁剪后存储图片的路径
    if not os.path.exists(cropImg):
        os.mkdir(cropImg)
    if not os.path.exists(cropAnno):
        os.mkdir(cropAnno)
    for each in tqdm(os.listdir(annotation)):
        # each = os.listdir(annotation)
        name = each.split('.')[0]
        origin_img_path = os.path.join(imgpath, name + '.jpg')
        origin_xml__path = os.path.join(annotation, name + '.xml')
        crop_img_path = os.path.join(cropImg, name)
        crop_xml__path = os.path.join(cropAnno, name)
        # tree = ET.parse(origin_xml__path)
        # root = tree.getroot()
        crop_dataset(origin_img_path, output_shape, origin_xml__path, crop_xml__path, crop_img_path, stride)
