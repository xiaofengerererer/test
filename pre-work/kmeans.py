# kmeans
import os
from tqdm import tqdm
from lxml import etree
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr

class VOCDataSet(object):
    def __init__(self, voc_root, year="2012", txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        self.root = os.path.join(voc_root, "../VOCdevkit", f"VOC{year}")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        # txt_path1 = os.path.join(self.root, "YOLOLabels")
        # txt_path = os.path.join(self.root, "YOLOLabels", txt_name)
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
class KMeans:
    dataset = []
    center_pointset = []
    K = 0

    def __init__(self, dataset, center_pointset, K):
        self.dataset = dataset
        self.center_pointset = center_pointset
        self.K = K

    def euclidean_distance(self, point1, point2):
        return math.sqrt(pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2))

    def set_euclidean_distance(self, set1, set2):
        if len(set1) == 0 or len(set2) == 0:
            return 1
        flag = 0
        for i in range(len(set1)):
            if self.euclidean_distance(set1[i], set2[i]) != 0:
                flag = 1
                break
        return flag

    def find_center_point(self, list):
        xsum = 0
        ysum = 0
        length = len(list)
        for data in list:
            xsum += data[0]
            ysum += data[1]
        return [xsum // length, ysum // length]

    def find_cluster_by_kmeans(self):
        kmeans_clusters = []
        count = 0
        old_center_pointset = self.center_pointset
        new_center_pointset = []
        flag = self.set_euclidean_distance(old_center_pointset, new_center_pointset)
        while count < 50 and flag != 0:
            if count != 0:
                old_center_pointset = new_center_pointset
            kmeans_clusters = [[] for _ in range(self.K)]
            for data in self.dataset:
                dist = []
                for i in range(len(old_center_pointset)):
                    distance = self.euclidean_distance(data, old_center_pointset[i])
                    dist.append(distance)
                kmeans_clusters[dist.index(min(dist))].append(data.tolist())
            count += 1
            new_center_pointset = []
            for cluster in kmeans_clusters:
                new_center_pointset.append(self.find_center_point(cluster))
            flag = self.set_euclidean_distance(old_center_pointset, new_center_pointset)
            print("更新后的中心点集：", end=" ")
            print(new_center_pointset)
        return new_center_pointset, kmeans_clusters

def anchor_fitness(k: np.ndarray, wh0: np.ndarray, thr: float):  # mutation fitness
    r = wh0[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr


class Visualization:
    center_points = []
    kmeans_cluster = []

    def __init__(self, center_points, kmeans_cluster):
        self.center_points = center_points
        self.kmeans_cluster = kmeans_cluster

    def format_point(self):
        lenth = len(self.kmeans_cluster)
        x_center_point = []
        y_center_point = []
        x_points = [[] for _ in range(lenth)]
        y_points = [[] for _ in range(lenth)]
        for center_point in self.center_points:
            x_center_point.append(center_point[0])
            y_center_point.append(center_point[1])
        for points in range(lenth):
            for point in self.kmeans_cluster[points]:
                x_points[points].append(point[0])
                y_points[points].append(point[1])
        return x_center_point, y_center_point, x_points, y_points

    def visual(self):
        x_center_point, y_center_point, x_point, y_point = self.format_point()
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'darkgreen', 'darkorange', 'gold', 'm','k','cyan','darkblue','aliceblue','aqua','beige']
        for i in range(len(x_point)):
            ax.scatter(x_point[i], y_point[i], c=colors[i])
        ax.scatter(x_center_point, y_center_point, marker='*', s=50, c='black')
        plt.show()

class Canopy:
    dataset = []
    t1 = 0
    t2 = 0

    def __init__(self, dataset, t1, t2):
        self.dataset = dataset
        self.t1 = t1
        self.t2 = t2

    def euclidean_distance(self, point1, point2):
        return math.sqrt(pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2))

    def get_index(self):
        return np.random.randint(len(self.dataset))

    def find_cluster_by_canopy(self):
        canopy_cluster = []
        while(len(self.dataset) != 0):
            center_set = []
            delete_set = []
            index = self.get_index()
            center_point = self.dataset[index]
            self.dataset = np.delete(self.dataset, index, 0)
            for i in range(len(self.dataset)):
                point = self.dataset[i]
                distance = self.euclidean_distance(point, center_point)
                if distance < self.t1:
                    center_set.append(point)
                if distance < self.t2:
                    delete_set.append(i)
            self.dataset = np.delete(self.dataset, delete_set, 0)
            canopy_cluster.append((center_point, center_set))
            canopy_cluster = [cluster for cluster in canopy_cluster if len(cluster[1]) > 1]
        return canopy_cluster

