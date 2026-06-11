import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


# =========================
# 设置中文显示
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 生成样本数据
# =========================

np.random.seed(42)

X = np.random.rand(100, 1) - 0.5

y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)


# 为了画平滑曲线，对 X 排序
X_new = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)


# =========================
# 2. 第一棵树拟合原始数据
# =========================

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y1_pred = tree_reg1.predict(X)
resid1 = y - y1_pred


# =========================
# 3. 第二棵树拟合第一次残差
# =========================

tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, resid1)

resid1_pred = tree_reg2.predict(X)
resid2 = resid1 - resid1_pred


# =========================
# 4. 第三棵树拟合第二次残差
# =========================

tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, resid2)

resid2_pred = tree_reg3.predict(X)
resid3 = resid2 - resid2_pred


# =========================
# 5. 手动累加前三棵树的预测结果
# =========================

manual_pred = (
    tree_reg1.predict(X_new)
    + tree_reg2.predict(X_new)
    + tree_reg3.predict(X_new)
)


# =========================
# 6. 使用 sklearn 的 GradientBoostingRegressor
# =========================

gbrt_3 = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1,
    random_state=42
)

gbrt_3.fit(X, y)

gbrt_3_pred = gbrt_3.predict(X_new)


# =========================
# 7. 使用 200 棵树
# =========================

gbrt_200 = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

gbrt_200.fit(X, y)

gbrt_200_pred = gbrt_200.predict(X_new)

resid_200 = y - gbrt_200.predict(X)


# =========================
# 8. 绘图：所有图放在一起
# =========================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

# 图1：原始数据
axes[0, 0].scatter(
    X[:, 0],
    y,
    c=y,
    cmap='viridis',
    s=45,
    edgecolor='black',
    alpha=0.8
)

axes[0, 0].set_title('图6-13 数据散点图')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('y')
axes[0, 0].grid(alpha=0.3)


# 图2：第一次拟合与第一次残差
axes[0, 1].scatter(
    X[:, 0],
    y,
    color='skyblue',
    edgecolor='black',
    s=45,
    alpha=0.8,
    label='训练集'
)

axes[0, 1].plot(
    X_new[:, 0],
    tree_reg1.predict(X_new),
    color='red',
    linewidth=2,
    label='h0(x)'
)

axes[0, 1].set_title('图6-14 第一次拟合')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)


axes[0, 2].scatter(
    X[:, 0],
    resid1,
    color='orange',
    edgecolor='black',
    s=45,
    alpha=0.8,
    label='残差1'
)

axes[0, 2].plot(
    X_new[:, 0],
    tree_reg2.predict(X_new),
    color='green',
    linewidth=2,
    label='h1(x)'
)

axes[0, 2].axhline(
    y=0,
    color='black',
    linestyle='--',
    linewidth=1
)

axes[0, 2].set_title('第一次残差图')
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('残差1')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)


# 图4：第二次累加拟合
second_pred = tree_reg1.predict(X_new) + tree_reg2.predict(X_new)

axes[1, 0].scatter(
    X[:, 0],
    y,
    color='skyblue',
    edgecolor='black',
    s=45,
    alpha=0.8,
    label='训练集'
)

axes[1, 0].plot(
    X_new[:, 0],
    second_pred,
    color='red',
    linewidth=2,
    label='h0(x)+h1(x)'
)

axes[1, 0].set_title('图6-15 第二次拟合')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)


# 图5：第三次累加拟合
axes[1, 1].scatter(
    X[:, 0],
    y,
    color='skyblue',
    edgecolor='black',
    s=45,
    alpha=0.8,
    label='训练集'
)

axes[1, 1].plot(
    X_new[:, 0],
    manual_pred,
    color='red',
    linewidth=2,
    label='h0(x)+h1(x)+h2(x)'
)

axes[1, 1].set_title('图6-16 第三次拟合')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('y')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)


# 图6：200棵树结果
axes[1, 2].scatter(
    X[:, 0],
    y,
    color='skyblue',
    edgecolor='black',
    s=45,
    alpha=0.8,
    label='训练集'
)

axes[1, 2].plot(
    X_new[:, 0],
    gbrt_200_pred,
    color='red',
    linewidth=2,
    label='集合预测'
)

axes[1, 2].set_title('图6-18 200棵决策树结果')
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('y')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)


plt.suptitle('Gradient Boosting 梯度提升回归过程', fontsize=18)
plt.tight_layout()
plt.show()


# =========================
# 9. 单独绘制 200 棵树残差图
# =========================

order = np.argsort(X[:, 0])

plt.figure(figsize=(10, 5))

plt.scatter(
    X[:, 0],
    resid_200,
    color='orange',
    edgecolor='black',
    s=45,
    alpha=0.8,
    label='残差200'
)

plt.plot(
    X[order, 0],
    resid_200[order],
    color='blue',
    linewidth=1.5,
    label='残差变化'
)

plt.axhline(
    y=0,
    color='black',
    linestyle='--',
    linewidth=1
)

plt.title('200棵树后的残差图')
plt.xlabel('X')
plt.ylabel('残差')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =========================
# 10. 输出结果
# =========================

print('前三棵树手动累加预测完成')
print('GradientBoostingRegressor 3棵树预测完成')
print('GradientBoostingRegressor 200棵树预测完成')

print('\n200棵树后残差均值：')
print(np.mean(resid_200))

print('\n200棵树后残差标准差：')
print(np.std(resid_200))