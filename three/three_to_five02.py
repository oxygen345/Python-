# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# =========================
# 1. 生成回归数据
# =========================

np.random.seed(0)

# 生成 40 个样本，每个样本只有 1 个特征
X = np.sort(6 * np.random.rand(40, 1) - 3, axis=0)

# 生成目标值 y
y = np.sin(X).ravel()

# 添加噪声
y += 0.3 * np.random.randn(40)


# =========================
# 2. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0
)


# =========================
# 3. 生成画图用的横坐标
# =========================

line = np.linspace(-3, 3, 1000).reshape(-1, 1)


# =========================
# 4. 分别绘制 K=1、K=3、K=9 的回归效果
# =========================

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

for k in [1, 3, 9]:
    reg = KNeighborsRegressor(n_neighbors=k)

    # 训练模型
    reg.fit(X_train, y_train)

    # 预测曲线
    y_line_pred = reg.predict(line)

    print(f"\n{k} 近邻回归")
    print("训练集分数：%.2f" % reg.score(X_train, y_train))
    print("测试集分数：%.2f" % reg.score(X_test, y_test))

    plt.figure(figsize=(7, 5))

    # 预测曲线
    plt.plot(line, y_line_pred, label='预测模型')

    # 训练数据
    plt.scatter(
        X_train,
        y_train,
        marker='^',
        label='训练数据/目标'
    )

    # 测试数据
    plt.scatter(
        X_test,
        y_test,
        marker='v',
        label='测试数据/目标'
    )

    plt.xlabel("特征")
    plt.ylabel("目标")
    plt.title(
        f"{k} 近邻\n训练分数：{reg.score(X_train, y_train):.2f}  "
        f"测试分数：{reg.score(X_test, y_test):.2f}"
    )

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()