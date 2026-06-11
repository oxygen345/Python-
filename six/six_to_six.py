import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from sklearn.ensemble import ExtraTreesRegressor as ET
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import AdaBoostRegressor as ADA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# =========================
# 设置中文显示
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 读取 Boston 数据集
# =========================
boston = fetch_openml(name='boston', version=1, as_frame=True)

X = boston.data
Y = boston.target

# 转成数值型，避免 category 报错
X = X.apply(pd.to_numeric, errors='coerce')
Y = pd.to_numeric(Y, errors='coerce')

print("Boston数据前5行：")
print(X.head())


# =========================
# 2. 划分训练集和测试集
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=123
)

# 转成 numpy，方便后续索引
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print("\n训练样例数：", X_train.shape[0])
print("测试样例数：", X_test.shape[0])
print("X_train形状：", X_train.shape)
print("Y_train形状：", Y_train.shape)


# =========================
# 3. 定义第一层模型
# =========================
models = [
    GBDT(n_estimators=100, random_state=123),
    RF(n_estimators=100, random_state=123),
    ET(n_estimators=100, random_state=123),
    ADA(n_estimators=100, random_state=123)
]

model_names = ['GBDT', 'RandomForest', 'ExtraTrees', 'AdaBoost']


# =========================
# 4. 构建 Stacking 第一层特征
# =========================
n_models = len(models)
n_folds = 10

X_train_stack = np.zeros((X_train.shape[0], n_models))
X_test_stack = np.zeros((X_test.shape[0], n_models))

kf = KFold(n_splits=n_folds, shuffle=True, random_state=123)

for i, model in enumerate(models):
    X_test_n_fold = np.zeros((X_test.shape[0], n_folds))

    for j, (train_index, valid_index) in enumerate(kf.split(X_train)):
        tr_x = X_train[train_index]
        tr_y = Y_train[train_index]
        val_x = X_train[valid_index]

        model.fit(tr_x, tr_y)

        # 训练集的第 i 列：用该折模型预测验证折
        X_train_stack[valid_index, i] = model.predict(val_x)

        # 测试集：每折都预测一次，最后取均值
        X_test_n_fold[:, j] = model.predict(X_test)

    X_test_stack[:, i] = X_test_n_fold.mean(axis=1)

print("\nX_train_stack形状：", X_train_stack.shape)
print("X_test_stack形状：", X_test_stack.shape)


# =========================
# 5. 第二层模型
# =========================
model_second = LinearRegression()
model_second.fit(X_train_stack, Y_train)
pred = model_second.predict(X_test_stack)


# =========================
# 6. 模型评估
# =========================
r2 = r2_score(Y_test, pred)
mae = mean_absolute_error(Y_test, pred)

print("\n模型评估结果：")
print("R2:", r2)
print("平均绝对误差 MAE:", mae)

print("\n部分真实值与预测值：")
for i in range(min(20, len(Y_test))):
    print("真实值: %.6f, 预测值: %.6f" % (Y_test[i], pred[i]))


# =========================
# 7. 图放在一起显示
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- 左图：真实值与预测值折线图 ----
axes[0].plot(
    Y_test,
    color='blue',
    marker='o',
    linestyle='-',
    linewidth=1.5,
    markersize=4,
    label='真实值'
)

axes[0].plot(
    pred,
    color='red',
    marker='s',
    linestyle='--',
    linewidth=1.5,
    markersize=4,
    label='预测值'
)

axes[0].set_title('真实值与预测值对比')
axes[0].set_xlabel('样本编号')
axes[0].set_ylabel('房价')
axes[0].legend()
axes[0].grid(alpha=0.3)


# ---- 右图：真实值-预测值散点图 ----
axes[1].scatter(
    Y_test,
    pred,
    color='purple',
    alpha=0.7,
    edgecolors='black',
    label='预测散点'
)

min_val = min(Y_test.min(), pred.min())
max_val = max(Y_test.max(), pred.max())

axes[1].plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linestyle='--',
    linewidth=2,
    label='理想预测线'
)

axes[1].set_title('真实值与预测值散点图')
axes[1].set_xlabel('真实值')
axes[1].set_ylabel('预测值')
axes[1].legend()
axes[1].grid(alpha=0.3)


# 总标题
fig.suptitle(
    'Stacking模型回归效果图\nR2=%.4f    MAE=%.4f' % (r2, mae),
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()