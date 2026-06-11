# -*- coding: utf-8 -*-

# 1. 导入需要的库
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# 2. 加载鸢尾花数据集
iris = datasets.load_iris()

# 只取两个特征：花瓣长度、花瓣宽度
X = iris.data[:, [2, 3]]

# 标签：0、1、2，分别表示三种鸢尾花
y = iris.target

print("Class labels:", np.unique(y))


# 3. 划分训练集和测试集
# test_size=0.3 表示测试集占 30%，训练集占 70%
# stratify=y 表示按照类别比例划分，保证每一类样本数量均衡
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y
)


# 4. 查看每一类样本数量
print("y 的标签计数:", np.bincount(y))
print("y_train 的标签计数:", np.bincount(y_train))
print("y_test 的标签计数:", np.bincount(y_test))


# 5. 数据标准化
# 标准化公式：标准化后的值 = (原始值 - 均值) / 标准差
sc = StandardScaler()

# 只在训练集上计算均值和标准差
sc.fit(X_train)

# 用训练集得到的均值和标准差分别标准化训练集和测试集
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# 6. 创建并训练感知器模型
ppn = Perceptron(
    max_iter=40,
    eta0=0.1,
    random_state=1
)

# 训练模型
ppn.fit(X_train_std, y_train)


# 7. 使用测试集进行预测
y_pred = ppn.predict(X_test_std)

# 统计错误分类的样本数量
print("错误分类的样本: %d" % (y_test != y_pred).sum())


# 8. 计算分类准确率
print("准确性: %.2f" % accuracy_score(y_test, y_pred))


# 9. 定义绘制决策区域的函数
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """
    绘制分类模型的决策区域

    参数说明：
    X：特征数据
    y：标签
    classifier：训练好的分类器
    test_idx：测试集样本的索引，用于在图中突出显示
    resolution：网格点间隔，越小图越细腻
    """

    # 设置不同类别的标记和颜色
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    # 根据类别数量选择颜色
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 找到两个特征的最大值和最小值
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 生成网格点
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 对网格中每一个点进行预测
    Z = classifier.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T
    )

    # 把预测结果变回网格形状
    Z = Z.reshape(xx1.shape)

    # 绘制决策区域
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    # 设置坐标轴范围
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制每一类样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'
        )

    # 突出显示测试集样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c='none',
            edgecolor='black',
            alpha=1.0,
            linewidth=1,
            marker='o',
            s=100,
            label='测试集'
        )


# 10. 合并训练集和测试集
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# 11. 绘制模型决策区域
plot_decision_regions(
    X=X_combined_std,
    y=y_combined,
    classifier=ppn,
    test_idx=range(105, 150)
)

# 设置中文字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 设置坐标轴名称
plt.xlabel('花瓣长度 [标准化]')
plt.ylabel('花瓣宽度 [标准化]')

# 显示图例
plt.legend(loc='upper left')

# 自动调整布局
plt.tight_layout()

# 显示图像
plt.show()