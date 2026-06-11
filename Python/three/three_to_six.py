# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


# =========================
# 1. 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei']      # 显示中文
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号


# =========================
# 2. 生成网格点函数
# =========================

def make_meshgrid(x, y, h=0.02):
    """
    根据两个特征生成网格点，用于绘制分类区域
    """

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    return xx, yy


# =========================
# 3. 绘制分类区域函数
# =========================

def plot_test_results(ax, clf, xx, yy, **params):
    """
    对网格点进行预测，并绘制分类区域
    """

    # 对网格中每一个点进行预测
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 改变预测结果的形状，使其和网格一致
    Z = Z.reshape(xx.shape)

    # 绘制分类区域
    ax.contourf(xx, yy, Z, **params)


# =========================
# 4. 加载 Iris 鸢尾花数据集
# =========================

iris = datasets.load_iris()

# 只使用前两个特征，方便画二维图
X = iris.data[:, :2]

# 样本标签
y = iris.target


# =========================
# 5. 创建并训练高斯朴素贝叶斯分类器
# =========================

clf = GaussianNB()

clf.fit(X, y)


# =========================
# 6. 绘图
# =========================

title = "高斯朴素贝叶斯分类器"

fig, ax = plt.subplots(figsize=(6, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)


# 分别取两个特征
X0, X1 = X[:, 0], X[:, 1]

# 生成网格点
xx, yy = make_meshgrid(X0, X1)


# 绘制分类区域
plot_test_results(
    ax,
    clf,
    xx,
    yy,
    cmap=plt.cm.coolwarm,
    alpha=0.8
)


# 绘制样本点
ax.scatter(
    X0,
    X1,
    c=y,
    cmap=plt.cm.coolwarm,
    s=20,
    edgecolors='k'
)


# 设置坐标轴范围
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())


# 设置坐标轴标签
ax.set_xlabel('x1')
ax.set_ylabel('x2')


# 去掉刻度
ax.set_xticks(())
ax.set_yticks(())


# 设置标题
ax.set_title(title)


# 显示图像
plt.show()