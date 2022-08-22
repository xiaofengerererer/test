import config
import cut_image
import joint_image
import draw_box
import os
import txt_to_yolo
import get_xml_data

def is_exists(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print('创建目录：', path.split('/')[-1])
        else:
            print(path.split('/')[-1], '目录已存在')

def main(config):
    config = config.Config
    ROOT_data = config.ROOT_DATA_PATH
    train_label_path = config.train_label_path
    train_images_path = config.train_images_path
    train_small_images_path = config.train_small_images_path
    mix_percent = config.mix_percent
    cut_row_num = config.cut_row_num
    cut_col_num = config.cut_col_num

    is_exists([ROOT_data, train_label_path, train_images_path, train_small_images_path])
    #
    # get_xml_data.main()
    cut_image.cut_image_main(train_images_path, train_small_images_path, cut_row_num, cut_col_num, mix_percent, save_mix=True)

    txt_to_yolo.txt_to_yolo()

    print("it's finished! Please checking the file!")
    return
main(config)
