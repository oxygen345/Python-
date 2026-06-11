from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 例5-6 使用 sklearn 分离半月形
# =========================

# 创建二维数据集，其中100个样本组成两个半月形
X, y = make_moons(
    n_samples=100,
    random_state=123
)


# =========================
# 1. 使用 KernelPCA 进行非线性降维
# =========================

scikit_kpca = KernelPCA(
    n_components=2,
    kernel='rbf',
    gamma=15
)

X_skernpca = scikit_kpca.fit_transform(X)


# =========================
# 2. 绘图：原始数据 + KPCA分离效果
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：原始半月形数据
axes[0].scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='red',
    marker='^',
    alpha=0.7,
    label='类别0'
)

axes[0].scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='blue',
    marker='o',
    alpha=0.7,
    label='类别1'
)

axes[0].set_title('原始半月形数据')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].legend()
axes[0].grid(alpha=0.3)


# 右图：KernelPCA 分离效果
axes[1].scatter(
    X_skernpca[y == 0, 0],
    X_skernpca[y == 0, 1],
    color='red',
    marker='^',
    alpha=0.7,
    label='类别0'
)

axes[1].scatter(
    X_skernpca[y == 1, 0],
    X_skernpca[y == 1, 1],
    color='blue',
    marker='o',
    alpha=0.7,
    label='类别1'
)

axes[1].set_title('图5-21 sklearn分离半月形效果')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('KernelPCA 分离半月形数据', fontsize=16)
plt.tight_layout()
plt.show()