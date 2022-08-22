
class Config:
    """
    General configuration parent class
    """
    # 切割成的行列数
    cut_row_num = 4
    cut_col_num = 4
    mix_percent = 0.2   # 混合比例
    ROOT_DATA_PATH = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007'   # 数据根目录

    # prework parameters
    train_label_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\YOLOLabels'
    train_images_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\JPEGImages'   # 原始图片路径
    train_small_images_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\JPEGImages1'   # 切割好的小图位置

    # detect work parameters
    detect_images_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\data\images'
    detect_small_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\data\small_images'
    detect_box_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\runs\detect\res'   # detect 检测后标注框
    detect_joint_labels_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\runs\detect\res'    # joint_image 融合后的 label for src_images
    detect_small_images_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\data\small_images_targets'  # 检测好目标的小图位置
    detect_small_labels_path = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\data\small_images_targets\labels'  # 检测好的小图label数据

