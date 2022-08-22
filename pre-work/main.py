import random
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans
import os
from lxml import etree
import matplotlib.pyplot as plt

from yolo_kmeans import k_means, wh_iou
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
    # print(k[None].shape)
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr


def main(img_size=640, n=12, thr=0.25, gen=1000):
    # 从数据集中读取所有图片的wh以及对应bboxes的wh
    dataset = VOCDataSet(voc_root="../", year="2007", txt_name="train.txt")
    im_wh, boxes_wh = dataset.get_info()

    # 最大边缩放到img_size
    im_wh = np.array(im_wh, dtype=np.float32)
    shapes = img_size * im_wh / im_wh.max(1, keepdims=True)
    wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])  # wh
    # print(wh0.shape)
    # plt.scatter(wh0[:,0],wh0[:,1])  # 标签 即为点代表的意思
    # plt.show()  # 显示所绘图形
    for i in range(12,13):
        k = k_means(wh0, i)

        # 按面积排序
        k = k[np.argsort(k.prod(1))]  # sort small to large
        f, bpr = anchor_fitness(k, wh0, thr)
        print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
        print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")
    # 画图
    print(k)
    plt.scatter(wh0[:,0],wh0[:,1])
    # 画聚类中心
    plt.scatter(k[:,0],k[:,1],marker='*',s=60)
    # for i in range(k):
    #     plt.annotate('中心'+str(i + 1),(Muk[i,0],Muk[i,1]))
    plt.show()

if __name__ == "__main__":
    main()
