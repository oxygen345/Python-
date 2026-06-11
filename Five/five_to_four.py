import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from matplotlib.ticker import FormatStrFormatter

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 创建半月形数据
# =========================

X, y = make_moons(
    n_samples=100,
    random_state=123
)


# =========================
# 2. 标准 PCA 降维
# =========================

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)


# =========================
# 3. 手写 RBF 核 PCA
# =========================

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF 核 PCA
    X：输入数据
    gamma：RBF核参数
    n_components：主成分数量
    """

    # 计算样本之间的欧式距离平方
    sq_dist = pdist(X, 'sqeuclidean')

    # 转换成方阵
    mat_sq_dists = squareform(sq_dist)

    # 计算 RBF 核矩阵
    K = np.exp(-gamma * mat_sq_dists)

    # 核矩阵居中
    N = K.shape[0]
    one_n = np.ones((N, N)) / N

    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 求特征值和特征向量
    eigvals, eigvecs = eigh(K)

    # 取最大的 n_components 个特征向量
    X_pc = np.column_stack(
        [eigvecs[:, -i] for i in range(1, n_components + 1)]
    )

    return X_pc


X_kpca = rbf_kernel_pca(
    X,
    gamma=15,
    n_components=2
)


# =========================
# 4. 三张图放在同一个窗口中
# =========================

fig, ax = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 4)
)

# 原始半月形数据
ax[0].scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='red',
    marker='^',
    alpha=0.7,
    label='类别0'
)

ax[0].scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='blue',
    marker='o',
    alpha=0.7,
    label='类别1'
)

ax[0].set_title('图5-15 创建的半月形')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].legend()


# PCA 降维效果
ax[1].scatter(
    X_spca[y == 0, 0],
    X_spca[y == 0, 1],
    color='red',
    marker='^',
    alpha=0.7,
    label='类别0'
)

ax[1].scatter(
    X_spca[y == 1, 0],
    X_spca[y == 1, 1],
    color='blue',
    marker='o',
    alpha=0.7,
    label='类别1'
)

ax[1].scatter(
    X_spca[y == 0, 0],
    np.zeros((50, 1)) + 0.02,
    color='red',
    marker='^',
    alpha=0.5
)

ax[1].scatter(
    X_spca[y == 1, 0],
    np.zeros((50, 1)) - 0.02,
    color='blue',
    marker='o',
    alpha=0.5
)

ax[1].set_title('图5-16 PCA降维效果')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].legend()


# KPCA 降维效果
ax[2].scatter(
    X_kpca[y == 0, 0],
    X_kpca[y == 0, 1],
    color='red',
    marker='^',
    alpha=0.7,
    label='类别0'
)

ax[2].scatter(
    X_kpca[y == 1, 0],
    X_kpca[y == 1, 1],
    color='blue',
    marker='o',
    alpha=0.7,
    label='类别1'
)

ax[2].scatter(
    X_kpca[y == 0, 0],
    np.zeros((50, 1)) + 0.02,
    color='red',
    marker='^',
    alpha=0.5
)

ax[2].scatter(
    X_kpca[y == 1, 0],
    np.zeros((50, 1)) - 0.02,
    color='blue',
    marker='o',
    alpha=0.5
)

ax[2].set_title('图5-17 KPCA降维效果')
ax[2].set_xlabel('PC1')
ax[2].set_ylabel('PC2')
ax[2].set_ylim([-1, 1])
ax[2].set_yticks([])
ax[2].legend()

# 坐标格式
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[2].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

plt.suptitle('KPCA 与 PCA 分离半月形数据对比', fontsize=16)
plt.tight_layout()
plt.show()