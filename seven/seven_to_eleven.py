import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 数据集
# =========================

x = np.array(list(range(1, 11))).reshape(-1, 1)

y = np.array([
    5.56, 5.70, 5.91, 6.40, 6.80,
    7.05, 8.90, 8.70, 9.00, 9.05
]).ravel()


# =========================
# 2. 建立模型
# =========================

# 决策树回归，最大深度为1
model1 = DecisionTreeRegressor(max_depth=1, random_state=0)

# 决策树回归，最大深度为3
model2 = DecisionTreeRegressor(max_depth=3, random_state=0)

# 线性回归
model3 = linear_model.LinearRegression()


# =========================
# 3. 训练模型
# =========================

model1.fit(x, y)
model2.fit(x, y)
model3.fit(x, y)


# =========================
# 4. 预测
# =========================

X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]

y_1 = model1.predict(X_test)
y_2 = model2.predict(X_test)
y_3 = model3.predict(X_test)


# =========================
# 5. 绘图
# =========================

plt.figure(figsize=(9, 6))

# 原始数据点
plt.scatter(
    x,
    y,
    s=40,
    edgecolor='black',
    c='darkorange',
    label='data'
)

# max_depth=1 的决策树
plt.plot(
    X_test,
    y_1,
    color='cornflowerblue',
    label='max_depth=1',
    linewidth=2
)

# max_depth=3 的决策树
plt.plot(
    X_test,
    y_2,
    color='yellowgreen',
    label='max_depth=3',
    linewidth=2
)

# 线性回归
plt.plot(
    X_test,
    y_3,
    color='red',
    label='linear regression',
    linewidth=2
)

plt.xlabel('数据')
plt.ylabel('目标')
plt.title('图7-20 决策树回归')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()