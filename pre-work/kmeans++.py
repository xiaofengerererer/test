import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lxml import etree

def train_txt():
    """
    合并多个txt
    """

    # 获取目标文件夹的路径
    path = r"C:\Users\87991\Desktop\qxf\yolov5-5.0\VOCdevkit\VOC2007\YOLOLabels"
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(path)
    result = "train.txt"
    # 打开当前目录下的result.txt文件，如果没有则创建
    file = open(result, 'w+', encoding="utf-8")
    # 向文件中写入字符

    # 先遍历文件名
    for filename in filenames:
        filepath = path + '/'
        filepath = filepath + filename
        # 遍历单个文件，读取行数
        for line in open(filepath, encoding="utf-8"):
            file.writelines(line)
        # file.write('\n')
    # 关闭文件
    file.close()

def loadDataSet(fileName):
    '''
    加载测试数据集，返回一个列表，列表的元素是一个坐标
    '''
    dataList = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataList.append(fltLine)
    return dataList


def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set, k):
    """
    1.从数据集中返回 k 个对象可作为质心
    """

    # 将矩阵转为列表
    data_set = np.matrix.tolist(data_set)

    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers)  # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):  # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break

    cluster_centers = np.mat(cluster_centers)
    return cluster_centers


def kMeans(dataSet, k):
    '''
    2.KMeans算法，返回最终的质心坐标和每个点所在的簇
    '''
    m = np.shape(dataSet)[0]  # m表示数据集的长度（个数）
    clusterAssment = np.mat(np.zeros((m, 2)))

    centroids = kpp_centers(dataSet, k)  # 保存k个初始质心的坐标
    clusterChanged = True
    iterIndex = 1  # 迭代次数
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = np.linalg.norm(np.array(centroids[j, :]) - np.array(dataSet[i, :]))
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
            # print("第%d次迭代后%d个质心的坐标:\n%s" % (iterIndex, k, centroids))  # 第一次迭代的质心坐标就是初始的质心坐标
            iterIndex += 1
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    iter = (iterIndex - 1) / dataSet.shape[0]
    return centroids, clusterAssment, iter


def showCluster(dataSet, k, centroids, clusterAssment, iters):
    '''
    数据可视化,只能画二维的图（若是三维的坐标图则直接返回1）
    '''
    numSamples, dim = dataSet.shape
    if dim != 2:
        return 1

    mark = ['or', 'ob', 'og', 'ok', 'oy', 'om', 'oc', '^r', '+g', 'sb', 'dk', '<y', 'pm']

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    # mark = ['Pr', 'Pb', 'Pg', 'Pk', 'Py', 'Pm', 'Pc', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    # for i in range(k):
    #     plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.title('Number of Iterations: %d' % (iters))
    plt.show()
class VOCDataSet(object):
    def __init__(self, voc_root, year="2012", txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        # txt_path1 = os.path.join(self.root, "YOLOLabels")
        # txt_path = os.path.join(self.root, 'YOLOLabels', txt_name)
        txt_path = os.path.join(self.root, txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:

            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]
            # print(self.xml_list)

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

    def __len__(self):
        return len(self.xml_list)

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def get_info(self):
        im_wh_list = []
        boxes_wh_list = []
        for xml_path in tqdm(self.xml_list, desc="read data info."):
            # read xml
            with open(xml_path,encoding='utf-8') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str.encode('utf-8'))
            data = self.parse_xml_to_dict(xml)["annotation"]
            im_height = int(data["size"]["height"])
            im_width = int(data["size"]["width"])
            # print(data['object'])
            # 取出图片大小，计算anchor在图片中的比例
            wh = []
            if 'object' not in data:
                continue
            else:
                for obj in data["object"]:
                    xmin = float(obj["bndbox"]["xmin"])
                    xmax = float(obj["bndbox"]["xmax"])
                    ymin = float(obj["bndbox"]["ymin"])
                    ymax = float(obj["bndbox"]["ymax"])
                    wh.append([(xmax - xmin) / im_width, (ymax - ymin) / im_height])

            if len(wh) == 0:
                continue

            im_wh_list.append([im_width, im_height])
            boxes_wh_list.append(wh)
        # print(boxes_wh_list)
        return im_wh_list, boxes_wh_list

def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr

if __name__ == '__main__':
    # 加载数据集
    # 从数据集中读取所有图片的wh以及对应bboxes的wh
    img_size = 640
    dataset = VOCDataSet(voc_root="../", year="2007", txt_name="train.txt")
    im_wh, boxes_wh = dataset.get_info()
    print('get info is finished!')

    # 最大边缩放到img_size
    im_wh = np.array(im_wh, dtype=np.float32)
    shapes = img_size * im_wh / im_wh.max(1, keepdims=True)
    wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])  # wh

    # fitness = []
    # for i in range(1,20):
    k = 12  # 选定k值，也就是簇的个数（可以指定为其他数）
    cent, clust, iters = kMeans(wh0, k)
    print('kmeans is finished!')
    cent = cent.tolist()
    cent = np.array(cent)
    cent = cent[np.argsort(cent.prod(1))]  # sort small to large

    f, bpr = anchor_fitness(cent, wh0, thr=0.3)
    print('the performance id finished!')
        # fitness.append(f)
    # plt.plot(range(1,20),fitness)
    plt.show()
    print("kmeans: " + " ".join([f"[{float(i[0])}, {float(i[1])}]" for i in cent]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")
    # 数据可视化处理
    showCluster(wh0, k, cent, clust, iters)



