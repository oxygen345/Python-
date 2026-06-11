import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.RandomState(42)

# =========================
# 1. 创建训练数据
# =========================
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# =========================
# 2. 创建测试数据（新的正常观察）
# =========================
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# =========================
# 3. 生成异常点（新的异常观察）
# =========================
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# =========================
# 4. 拟合孤立森林模型
# =========================
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# =========================
# 5. 生成背景网格
# =========================
xx, yy = np.meshgrid(
    np.linspace(-5, 5, 200),
    np.linspace(-5, 5, 200)
)

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# =========================
# 6. 绘图
# =========================
plt.figure(figsize=(8, 6))
plt.title("孤立森林")

# 彩色背景
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# 训练观察：白色圆点
b1 = plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c='white',
    s=20,
    edgecolor='k',
    marker='o'
)

# 新的常规观察：绿色方块
b2 = plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c='green',
    s=20,
    edgecolor='k',
    marker='s'
)

# 新的异常观察：红色圆点
c = plt.scatter(
    X_outliers[:, 0], X_outliers[:, 1],
    c='red',
    s=20,
    edgecolor='k',
    marker='o'
)

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

plt.legend(
    [b1, b2, c],
    ["训练观察", "新的常规观察", "新的异常观察"],
    loc='upper left'
)

plt.show()