import os
import config

def txt_to_yolo(txt_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\YOLOLabels1',
                labels_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\YOLOLabels2'):
# 原txt文件位置

# 裁剪后label文件位置


    txt_names = os.listdir(txt_path)
    # txt_names = os.listdir(test_path)
    for txt_name in txt_names:
        f = open(txt_path + '\\' + txt_name, 'r')
        content = f.read().split('\n')[:-1]
        if content ==[]:
            continue
        f.close()
        # print(txt_name)
        child_type = []
        child_num = []
        child_name = []
        img_width = []
        img_height = []
        label_index = []
        child_xmin = []
        child_xmax = []
        child_ymin = []
        child_ymax = []

        # 读取数据
        for list in content:
            list = list.split(' ')
            child_type.append(list[0])
            child_num.append(list[1])
            child_name.append(list[2])
            img_width.append(list[3])
            img_height.append(list[4])
            label_index.append(list[5])
            child_xmin.append(list[6])
            child_xmax.append(list[7])
            child_ymin.append(list[8])
            child_ymax.append(list[9])
        # 处理不同宽高的图片，仅限于 分成 4*4 的， 混合度 20%
        child_height = int(img_height[0]) // 4
        child_width = int(img_width[0]) // 4
        mix_row_width = int(child_width * 0.2 * 2)
        mix_row_height = child_height
        mix_col_width = child_width
        mix_col_height = int(child_height * 0.2 * 2)
        row_width_step = int(child_width * 0.8)
        col_height_step = int(child_height * 0.8)


        # 根据child_name进行分类
        child_name_dict = {}
        for i in range(len(child_name)):
            if child_name[i] not in child_name_dict:
                child_name_dict[child_name[i]] = [[child_type[i], child_num[i], label_index[i], child_xmin[i], child_xmax[i], child_ymin[i], child_ymax[i]]]
            else:
                child_name_dict[child_name[i]].append([child_type[i], child_num[i], label_index[i], child_xmin[i], child_xmax[i], child_ymin[i], child_ymax[i]])
        # for key in child_name_dict:
        #     print(key, child_name_dict[key])

        # 写入文件
        # 需要处理不同位置的裁剪
        # type 0-不裁剪， 1-row 2-col
        for key in child_name_dict:
            content = ''
            child_file_name = key + '.txt'
            # print(key, child_name_dict[key])
            # todo info: type, num, label_index, xmin, xmax, ymin, ymax
            for info in child_name_dict[key]:
                # print(info)
                # num(0-15) 对应
                # (0, 0) (0, 1) (0, 2) (0, 3)
                # (1, 0) (1, 1) (1, 2) (1, 3)
                # (2, 0) (2, 1) (2, 2) (2, 3)
                # (3, 0) (3, 1) (3, 2) (3, 3)
                start_x = 0.0
                start_y = 0.0
                width = 0.0
                height = 0.0
                # 小块图像
                if info[0] == '0':
                    x = int(info[1]) // 4  #取整
                    y = int(info[1]) % 4  #取余
                    # print('-----------')
                    # print(x)
                    # print(y)
                    # print(info[1], info[2], info[3], info[4], info[5], info[6])
                    info[3] = int(info[3]) - x * child_width
                    info[4] = int(info[4]) - x * child_width
                    info[5] = int(info[5]) - y * child_height
                    info[6] = int(info[6]) - y * child_height
                    # print(info[3], info[4], info[5], info[6])
                    if int(info[3]) >= 0 and int(info[4]) >= 0 and int(info[5]) >= 0 and int(info[6]) >= 0:
                    # assert int(info[3]) >= 0 and int(info[4]) >= 0 and int(info[5]) >= 0 and int(info[6]) >= 0, '?小于0'
                        start_x = (int(info[3]) + int(info[4])) / 2 / child_width
                        start_y = (int(info[5]) + int(info[6])) / 2 / child_height
                        width = (int(info[4]) - int(info[3])) / child_width
                        height = (int(info[6]) - int(info[5])) / child_height
                    else:
                        continue
                # mix_row 图像
                elif info[0] == '1':
                    # print(info[2], info[3], info[4], info[5], info[6])
                    x = int(info[1]) % 3
                    y = int(info[1]) // 3
                    info[3] = int(info[3]) - (row_width_step + x * child_width)
                    info[4] = int(info[4]) - (row_width_step + x * child_width)
                    info[5] = int(info[5]) - y * child_height
                    info[6] = int(info[6]) - y * child_height
                    # print(info[3], info[4], info[5], info[6])
                    if int(info[3]) >= 0 and int(info[4]) >= 0 and int(info[5]) >= 0 and int(info[6]) >= 0:
                    # assert int(info[3]) >= 0 and int(info[4]) >= 0 and int(info[5]) >= 0 and int(info[6]) >= 0, '??小于0'
                        start_x = (int(info[3]) + int(info[4])) / 2 / mix_row_width
                        start_y = (int(info[5]) + int(info[6])) / 2 / mix_row_height
                        width = (int(info[4]) - int(info[3])) / mix_row_width
                        height = (int(info[6]) - int(info[5])) / mix_row_width
                    else:
                        continue
                # mix_col 图像
                elif info[0] == '2':
                    # print(info[1], info[3], info[4], info[5], info[6])
                    x = int(info[1]) % 4
                    y = int(info[1]) // 4
                    info[3] = int(info[3]) - x * child_width
                    info[4] = int(info[4]) - x * child_width
                    # print(col_height_step)
                    info[5] = int(info[5]) - (col_height_step + y * child_height)
                    info[6] = int(info[6]) - (col_height_step + y * child_height)
                    # print(info[3], info[4], info[5], info[6])
                    if int(info[3]) >= 0 and int(info[4]) >= 0 and int(info[5]) >= 0 and int(info[6]) >= 0:
                    # assert int(info[3]) >= 0 and int(info[4]) >= 0 and int(info[5]) >= 0 and int(info[6]) >= 0, '???小于0'
                        start_x = (int(info[3]) + int(info[4])) / 2 / mix_col_width
                        start_y = (int(info[5]) + int(info[6])) / 2 / mix_col_height
                        width = (int(info[4]) - int(info[3])) / mix_row_width
                        height = (int(info[6]) - int(info[5])) / mix_col_height
                    else:
                        continue
                # print(start_x, start_y, width, height)
                assert int(start_x) == 0 or int(start_y) == 0 or int(width) == 0 or int(height) == 0, \
                    'start_x, start_y, width, height 有为0'
                content += str(info[2]) + ' ' + str(start_x) + ' ' + str(start_y) + ' ' + str(width) + ' ' + str(height) + '\n'
            # print(content)
            with open(labels_path + '\\' + child_file_name, 'w') as f:
                f.write(content)
    print('finished the txt_to_yolo.py')
if __name__ == '__main__':
    txt_to_yolo()
