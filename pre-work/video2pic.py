# 导入所需要的库
import cv2
import numpy as np

# 定义保存图片函数
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)

# 读取视频文件
videoCapture = cv2.VideoCapture('./mask4.mp4')  #视频路径
# 通过摄像头的方式
# videoCapture=cv2.VideoCapture(1)
print(videoCapture)
# 读帧
success, frame = videoCapture.read()
print(success)
i = 0
timeF = 10 #视频帧率设置
j = 1992
while success:
    i = i + 1
    if (i % timeF == 0):
        j = j + 1
        save_image(frame, './output/', j)
        #图片保存目录，需提前建立好一个名为output的文件夹
        print('save image:', i)
    success, frame = videoCapture.read()
