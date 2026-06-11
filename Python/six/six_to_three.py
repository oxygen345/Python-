import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


# =========================
# 设置中文显示
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 构建数据集
# =========================

X, y = make_moons(
    n_samples=500,
    noise=0.30,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)


# =========================
# 2. 绘制决策边界函数
# =========================

def plot_decision_boundary(clf, X, y, axes=(-1.5, 2.5, -1.0, 1.5), alpha=0.2):
    x1s = np.linspace(axes[0], axes[1], 300)
    x2s = np.linspace(axes[2], axes[3], 300)

    x1, x2 = np.meshgrid(x1s, x2s)

    X_new = np.c_[x1.ravel(), x2.ravel()]

    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap([
        '#fafab0',
        '#9898ff'
    ])

    plt.contourf(
        x1,
        x2,
        y_pred,
        cmap=custom_cmap,
        alpha=alpha
    )

    plt.contour(
        x1,
        x2,
        y_pred,
        colors='black',
        linewidths=1,
        alpha=0.8
    )

    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color='orange',
        marker='o',
        alpha=0.6,
        label='类别0'
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='blue',
        marker='s',
        alpha=0.6,
        label='类别1'
    )

    plt.axis(axes)
    plt.xlabel('x1')
    plt.ylabel('x2')


# =========================
# 3. SVM 实现 AdaBoost 思想
# =========================

plt.figure(figsize=(14, 6))

learning_rate = 1

# 初始化样本权重，每个样本初始权重都为 1
sample_weights = np.ones(len(X_train))

for i in range(5):
    # 使用 SVM 作为基本分类器
    svm_clf = SVC(
        kernel='rbf',
        C=0.05,
        random_state=42
    )

    # 根据样本权重训练模型
    svm_clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    # 预测训练集
    y_pred = svm_clf.predict(X_train)

    # 被分错的样本权重增大
    sample_weights[y_pred != y_train] *= (1 + learning_rate)

    # 每一轮画一次决策边界
    plot_decision_boundary(
        svm_clf,
        X,
        y,
        alpha=0.2
    )

    # 在图中标注第几轮
    if i == 0:
        plt.text(-0.7, -0.65, '1', fontsize=14)
    elif i == 1:
        plt.text(-0.6, -0.10, '2', fontsize=14)
    elif i == 2:
        plt.text(-0.5, 0.10, '3', fontsize=14)
    elif i == 3:
        plt.text(-0.4, 0.55, '4', fontsize=14)
    elif i == 4:
        plt.text(-0.3, 0.90, '5', fontsize=14)


plt.title('图6-11 SVM实现AdaBoost分类效果')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()