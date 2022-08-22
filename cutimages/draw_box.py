# -*- coding: utf-8 -*-
import os
from PIL import Image
from PIL import ImageDraw, ImageFont
from cv2 import cv2


def draw_images(images_dir, txt_dir, box_dir, font_type_path):
    font = ImageFont.truetype(font_type_path, 50)
    if not os.path.exists(box_dir):
        os.makedirs(box_dir)
    # num = 0

    # 设置颜色
    # all_colors = ['red', 'green', 'yellow', 'blue', 'pink', 'black', 'skyblue', 'brown', 'orange', 'purple', 'gray',
    #               'lightpink', 'gold', 'brown', 'black']
    all_colors = ['red', 'green', 'yellow', 'blue', 'pink', 'black', 'skyblue', 'brown', 'orange', 'purple']
    colors = {}

    for file in os.listdir(txt_dir):
        print(file)
        image = os.path.splitext(file)[0].replace('xml', 'jpg') + '.jpg'
        # 转换成cv2读取，防止图片载入错误
        img = cv2.imread(images_dir + '/' + image)
        TURN = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(TURN)
        # img.show()

        if img.mode == "P":
            img = img.convert('RGB')

        w, h = img.size
        tag_path = txt_dir + '/' + file
        with open(tag_path) as f:
            for line in f:
                line_parts = line.split(' ')
                # 根据不同的 label 保存颜色
                if line_parts[0] not in colors.keys():
                    colors[line_parts[0]] = all_colors[len(colors.keys())]
                color = colors[line_parts[0]]

                draw = ImageDraw.Draw(img)
                x = (float(line_parts[1]) - 0.5 * float(line_parts[3])) * w
                y = (float(line_parts[2]) - 0.5 * float(line_parts[4])) * h
                xx = (float(line_parts[1]) + 0.5 * float(line_parts[3])) * w
                yy = (float(line_parts[2]) + 0.5 * float(line_parts[4])) * h
                draw.rectangle([x, y, xx, yy], fill=None, outline=color, width=5)
                # num += 1
            del draw
            img.save(box_dir + '/' + image)
        # print(file, num)
    # print(colors)


def draw_main(box_dir=r'G:\Unity\all_data\big_images_box',
              txt_dir=r'G:\Unity\all_data\labels',
              image_source_dir=r'G:\Unity\all_data\big_images'):
    font_type_path = 'C:/Windows/Fonts/simsun.ttc'
    print(f'标注框, 数据来源: {txt_dir}\n 被标注图片: {image_source_dir}\n 结果保存路径: {box_dir}')
    draw_images(image_source_dir, txt_dir, box_dir, font_type_path)


# draw_main()
