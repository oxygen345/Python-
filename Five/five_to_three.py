import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 例5-3 使用 sklearn 库实现 LDA
# =========================

# 生成1000个三维样本
x, y = make_classification(
    n_samples=1000,
    n_features=3,
    n_redundant=0,
    n_classes=3,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=0.5,
    random_state=10
)

# =========================
# PCA 降维
# =========================

model1 = PCA(n_components=2)
x1 = model1.fit_transform(x)

# =========================
# LDA 降维
# =========================

model2 = LinearDiscriminantAnalysis(n_components=2)
x2 = model2.fit_transform(x, y)


# =========================
# 三张图放在同一个窗口
# =========================

fig = plt.figure(figsize=(18, 5))

# 1. 三维散点图
ax1 = fig.add_subplot(1, 3, 1, projection='3d')

p1 = ax1.scatter(
    x[:, 0],
    x[:, 1],
    x[:, 2],
    c=y,
    cmap='viridis',
    s=20,
    alpha=0.8
)

ax1.set_title('图5-11 三维散点图')
ax1.set_xlabel('特征1')
ax1.set_ylabel('特征2')
ax1.set_zlabel('特征3')


# 2. PCA降维效果
ax2 = fig.add_subplot(1, 3, 2)

p2 = ax2.scatter(
    x1[:, 0],
    x1[:, 1],
    c=y,
    cmap='viridis',
    s=20,
    alpha=0.8
)

ax2.set_title('图5-12 PCA降维效果')
ax2.set_xlabel('第一主成分')
ax2.set_ylabel('第二主成分')
ax2.grid(alpha=0.3)


# 3. LDA降维效果
ax3 = fig.add_subplot(1, 3, 3)

p3 = ax3.scatter(
    x2[:, 0],
    x2[:, 1],
    c=y,
    cmap='viridis',
    s=20,
    alpha=0.8
)

ax3.set_title('图5-13 LDA降维效果')
ax3.set_xlabel('第一判别方向')
ax3.set_ylabel('第二判别方向')
ax3.grid(alpha=0.3)

# 添加统一颜色条
cbar = fig.colorbar(p3, ax=[ax1, ax2, ax3], shrink=0.75)
cbar.set_label('类别')

plt.suptitle('sklearn实现LDA与PCA降维效果对比', fontsize=16)
plt.show()