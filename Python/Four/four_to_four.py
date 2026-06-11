import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 例4-4 利用 DBSCAN 算法检测异常数据
# =========================

# 生成随机簇类数据，样本数为600，类别为5
X, y = make_blobs(
    random_state=170,
    n_samples=600,
    centers=5
)

# DBSCAN聚类
# 按照经验 minPts = 2 * 维度数
# 这里数据是二维，所以 min_samples = 4
dbscan = DBSCAN(
    eps=1,
    min_samples=4
)

clusters = dbscan.fit_predict(X)

# 性能评价指标 ARI
ari = adjusted_rand_score(y, clusters)
print("ARI=", round(ari, 2))


# =========================
# 绘图
# =========================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 左图：原始随机数据
axes[0].scatter(
    X[:, 0],
    X[:, 1],
    c='gray',
    s=20
)

axes[0].set_xlabel("特征0")
axes[0].set_ylabel("特征1")
axes[0].set_title("(a) 随机簇数据")


# 右图：DBSCAN聚类结果
axes[1].scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    cmap='plasma',
    s=20
)

axes[1].set_xlabel("特征0")
axes[1].set_ylabel("特征1")
axes[1].set_title("(b) 聚类结果")


plt.suptitle("图4-5 检测异常数据结果")
plt.tight_layout()
plt.show()