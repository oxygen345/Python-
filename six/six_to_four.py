import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

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

def plot_decision_boundary(clf, X, y, axes=(-1.5, 2.5, -1.0, 1.5)):
    x1s = np.linspace(axes[0], axes[1], 300)
    x2s = np.linspace(axes[2], axes[3], 300)

    x1, x2 = np.meshgrid(x1s, x2s)

    X_new = np.c_[x1.ravel(), x2.ravel()]

    y_pred = clf.predict(X_new).reshape(x1.shape)

    # 背景颜色
    custom_cmap = ListedColormap([
        '#FFF2B2',
        '#B7C9FF'
    ])

    plt.contourf(
        x1,
        x2,
        y_pred,
        cmap=custom_cmap,
        alpha=0.6
    )

    # 分类边界线
    plt.contour(
        x1,
        x2,
        y_pred,
        colors='black',
        linewidths=1,
        alpha=0.7
    )

    # 类别0
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color='orange',
        marker='o',
        alpha=0.7,
        label='类别0'
    )

    # 类别1
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='blue',
        marker='s',
        alpha=0.7,
        label='类别1'
    )

    plt.axis(axes)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(alpha=0.3)


# =========================
# 3. 使用 sklearn 实现 AdaBoost
# =========================

ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

ada_clf.fit(X_train, y_train)

y_pred = ada_clf.predict(X_test)

print('AdaBoost 测试集准确率：')
print(round(accuracy_score(y_test, y_pred), 3))


# =========================
# 4. 绘制 AdaBoost 分类效果
# =========================

plt.figure(figsize=(8, 6))

plot_decision_boundary(
    ada_clf,
    X,
    y
)

plt.title('图6-12 使用 sklearn 实现 AdaBoost 分类')
plt.show()