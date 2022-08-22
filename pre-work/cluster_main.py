from kmeans import *
import numpy as np

points = []
center_points = []
K = 0
img_size=640
n=12
thr=0.25
gen=1000
dataset = VOCDataSet(voc_root="./", year="2007", txt_name="train.txt")
im_wh, boxes_wh = dataset.get_info()
# 最大边缩放到img_size
im_wh = np.array(im_wh, dtype=np.float32)
# 按比例缩放
shapes = img_size * im_wh / im_wh.max(1, keepdims=True)
wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])  # wh
points = np.array(wh0)

# # -----------canopy---------------------
canopy = Canopy(points, t1=160, t2=90)
canopy_cluster = canopy.find_cluster_by_canopy()
for i in canopy_cluster:
    center_points.append(i[0].tolist())
K = len(center_points)
# -----------canopy---------------------
# K=12
# center_points = points[np.random.choice(points.shape[0], K, replace=False)]
# ----------k-means---------------------
kmeans = KMeans(points, center_points, K)
center_points, kmeans_cluster = kmeans.find_cluster_by_kmeans()
center_points = np.array(center_points)
f, bpr = anchor_fitness(center_points, wh0, thr)

print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in center_points]))
print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")

# print(center_points)

visual = Visualization(center_points, kmeans_cluster)
visual.visual()
