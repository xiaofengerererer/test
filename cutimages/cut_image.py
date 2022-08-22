import os
from cv2 import cv2
# import config

# mix_percent = 0.2  # 图片混合比例


def split_image(src_path, rownum, colnum, file, save_path, mix_percent, save_mix=True):
    # save_path+file+编号 组成保存的新文件名
    if src_path.split('.')[-1] == 'jpg':
        img = cv2.imread(src_path)
        # cv2.imwrite(path, img)
        size = img.shape[0:2]
        w = size[1]
        h = size[0]
        # print(file, w, h)
        # 每行的高度和每列的宽度
        row_height = h // rownum
        col_width = w // rownum
        num = 0
        for i in range(rownum):
            for j in range(colnum):
                new_path = save_path
                row_start = j * col_width
                row_end = (j + 1) * col_width
                col_start = i * row_height
                col_end = (i + 1) * row_height
                # print(row_start, row_end, col_start, col_end)
                # cv2图片： [高， 宽]
                child_img = img[col_start:col_end, row_start:row_end]
                new_path = save_path + '/' + file + '_' + str(num) + '.jpg'
                cv2.imwrite(new_path, child_img)
                num += 1
        if save_mix:
            img_mix(img, row_height, col_width, save_path, mix_percent, file)


def img_mix(img, row_height, col_width, save_path, mix_percent, file):
    mix_num = 3
    # 每行的高度和每列的宽度

    # 分割成4*4就是有
    # 4*3个行融合区域
    # 3*4个列融合区域
    # 一行的融合
    row = 0
    for i in range(mix_num + 1):
        mix_height_start = i * row_height
        mix_height_end = (i + 1) * row_height
        for j in range(mix_num):
            mix_row_path = save_path + '/' + file + '_mix_row_' + str(row) + '.jpg'
            mix_row_start = int(j * col_width + col_width * (1 - mix_percent))
            mix_row_end = int(mix_row_start + col_width * mix_percent * 2)
            # print(mix_height_start, mix_height_end, mix_row_start, mix_row_end)
            mix_row_img = img[mix_height_start:mix_height_end, mix_row_start:mix_row_end]
            cv2.imwrite(mix_row_path, mix_row_img)
            row += 1

    col = 0
    # 一列的融合
    for i in range(mix_num):
        mix_col_start = int(i * row_height + row_height * (1 - mix_percent))
        mix_col_end = int(mix_col_start + row_height * mix_percent * 2)
        for j in range(mix_num + 1):
            mix_col_path = save_path + '/' + file + '_mix_col_' + str(col) + '.jpg'
            mix_width_start = j * col_width
            mix_width_end = (j + 1) * col_width
            # print(mix_col_start, mix_col_end, mix_width_start, mix_width_end)
            mix_col_img = img[mix_col_start:mix_col_end, mix_width_start:mix_width_end]
            cv2.imwrite(mix_col_path, mix_col_img)
            col += 1


def cut_image_main(file_path=r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\JPEGImages',
                   save_path=r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\JPEGImages1',
                   cut_row_num=4, cut_col_num=4,mix_percent=0.2,
                   save_mix=True):
    print(f'begin to segmentate images, save mix') if save_mix else print(f'begin to cut images, no save mix')
    file_names = os.listdir(file_path)
    for file in file_names:
        src = file_path + '\\' + file
        split_image(src, cut_row_num, cut_col_num, file.split('.')[0], save_path, mix_percent, save_mix)
    print(f'finished the segmentation，the small pictures save in:{save_path}')

# cut_image_main()
