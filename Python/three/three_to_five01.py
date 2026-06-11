 # -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


# =========================
# 1. 加载鸢尾花数据集
# =========================

iris = datasets.load_iris()

# 取花瓣长度和花瓣宽度两个特征
X = iris.data[:, [2, 3]]
y = iris.target


# =========================
# 2. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y
)


# =========================
# 3. 交叉验证选择最优 K 值
# =========================

print("不同 K 值的交叉验证结果：")

all_scores = []

for k in range(3, 10):
    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn,
        X_train,
        y_train,
        cv=10
    )

    mean_score = scores.mean()
    std_score = scores.std()

    all_scores.append((k, mean_score))

    print(f"K={k}, 平均分数={mean_score:.4f}, 标准差={std_score:.4f}")


best_k = sorted(all_scores, key=lambda x: x[1], reverse=True)[0][0]

print("\n最优 K 值：", best_k)


# =========================
# 4. 使用最优 K 值训练模型
# =========================

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train, y_train)


# =========================
# 5. 测试集预测
# =========================

y_pred = knn.predict(X_test)

print("\n测试集真实标签：")
print(y_test)

print("\n测试集预测标签：")
print(y_pred)

print("\n错误分类的样本数：", (y_test != y_pred).sum())
print("测试集准确率：%.2f" % accuracy_score(y_test, y_pred))


# =========================
# 6. 绘制分类边界函数
# =========================

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'D', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    Z = classifier.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T
    )

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

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


# =========================
# 7. 绘制 K=1、K=3、K=9 的分类效果
# =========================

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

for k in [1, 3, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    plt.figure(figsize=(6, 4))

    plot_decision_regions(
        X_train,
        y_train,
        classifier=model
    )

    plt.xlabel("花瓣长度")
    plt.ylabel("花瓣宽度")
    plt.title(f"{k} 近邻分类效果")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()