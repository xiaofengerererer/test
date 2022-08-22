import os


def readname():
    filePath = r'C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\Annotations'
    name = os.listdir(filePath)
    return name


if __name__ == "__main__":
    name = readname()
    # print(name)
    for i in name:
        print(i.split('.')[0])
