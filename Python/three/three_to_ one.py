# -*- coding: utf-8 -*-

# 1. 导入库
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# 2. 加载鸢尾花数据集
iris = datasets.load_iris()

# 只取两个特征：花瓣长度、花瓣宽度
X = iris.data[:, [2, 3]]

# 标签：0、1、2
y = iris.target

print("Class labels:", np.unique(y))


# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y
)


# 4. 查看标签数量
print("y 的标签计数:", np.bincount(y))
print("y_train 的标签计数:", np.bincount(y_train))
print("y_test 的标签计数:", np.bincount(y_test))


# 5. 数据标准化
sc = StandardScaler()

# 只在训练集上计算均值和标准差
sc.fit(X_train)

# 标准化训练集和测试集
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# 6. 定义绘制决策区域函数
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'D', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 设置坐标范围
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 生成网格点
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # 对网格点进行预测
    Z = classifier.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T
    )

    Z = Z.reshape(xx1.shape)

    # 画分类背景区域
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    # 设置坐标轴范围
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 画样本点
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

    # 突出显示测试集
    if test_idx:
        X_test_show = X[test_idx, :]

        plt.scatter(
            X_test_show[:, 0],
            X_test_show[:, 1],
            c='none',
            edgecolor='black',
            alpha=1.0,
            linewidth=1,
            marker='o',
            s=100,
            label='测试集'
        )


# 7. 创建逻辑回归模型
lr = LogisticRegression(
    C=100.0,
    random_state=1
)


# 8. 训练模型
lr.fit(X_train_std, y_train)


# 9. 预测测试集
y_pred = lr.predict(X_test_std)

print("错误分类的样本:", (y_test != y_pred).sum())
print("准确性: %.2f" % accuracy_score(y_test, y_pred))


# 10. 合并训练集和测试集，方便画图
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# 11. 绘制决策区域
plot_decision_regions(
    X_combined_std,
    y_combined,
    classifier=lr,
    test_idx=range(105, 150)
)


# 12. 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 13. 设置图像信息
plt.xlabel('花瓣长度 [标准化]')
plt.ylabel('花瓣宽度 [标准化]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()