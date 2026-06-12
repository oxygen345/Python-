import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# 一、单输出回归
# =========================================================

rng = np.random.RandomState(1)

X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)

y = np.array([np.pi * np.sin(X).ravel()]).T

y += (0.5 - rng.rand(*y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=400,
    test_size=200,
    random_state=4
)

regr_rf_single = RandomForestRegressor(
    n_estimators=100,
    max_depth=30,
    random_state=2
)

regr_rf_single.fit(X_train, y_train.ravel())

y_rf_single = regr_rf_single.predict(X_test)

score_single = regr_rf_single.score(X_test, y_test.ravel())

print("单输出随机森林评价：", score_single)


# =========================================================
# 二、多输出回归
# =========================================================

rng = np.random.RandomState(1)

X2 = np.sort(200 * rng.rand(600, 1) - 100, axis=0)

y2 = np.array([
    np.pi * np.sin(X2).ravel(),
    np.pi * np.cos(X2).ravel()
]).T

y2 += (0.5 - rng.rand(*y2.shape))

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    train_size=400,
    test_size=200,
    random_state=4
)

max_depth = 30

regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=100,
        max_depth=max_depth,
        random_state=0
    )
)

regr_rf_multi = RandomForestRegressor(
    n_estimators=100,
    max_depth=max_depth,
    random_state=2
)

regr_multirf.fit(X2_train, y2_train)

regr_rf_multi.fit(X2_train, y2_train)

y_multirf = regr_multirf.predict(X2_test)

y_rf_multi = regr_rf_multi.predict(X2_test)

score_multirf = regr_multirf.score(X2_test, y2_test)

score_rf_multi = regr_rf_multi.score(X2_test, y2_test)

print("多输出随机森林评价：", score_multirf)
print("普通随机森林评价：", score_rf_multi)


# =========================================================
# 三、两个图放在同一个窗口
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

s = 50
a = 0.4


# =========================
# 左图：图7-21 单输出随机森林
# =========================

axes[0].scatter(
    X_test,
    y_test,
    edgecolor='k',
    c='navy',
    s=s,
    marker='s',
    alpha=a,
    label='Data'
)

axes[0].scatter(
    X_test,
    y_rf_single,
    edgecolor='k',
    c='cornflowerblue',
    s=s,
    marker='^',
    alpha=a,
    label='RF score=%.2f' % score_single
)

axes[0].set_xlim([-6, 6])
axes[0].set_xlabel('测试集')
axes[0].set_ylabel('目标')
axes[0].set_title('图7-21 随机森林和测试集')
axes[0].legend()
axes[0].grid(alpha=0.3)


# =========================
# 右图：图7-22 多输出随机森林
# =========================

axes[1].scatter(
    y2_test[:, 0],
    y2_test[:, 1],
    edgecolor='k',
    c='navy',
    s=s,
    marker='s',
    alpha=a,
    label='Data'
)

axes[1].scatter(
    y_multirf[:, 0],
    y_multirf[:, 1],
    edgecolor='k',
    c='cornflowerblue',
    s=s,
    alpha=a,
    label='Multi RF score=%.2f' % score_multirf
)

axes[1].scatter(
    y_rf_multi[:, 0],
    y_rf_multi[:, 1],
    edgecolor='k',
    c='red',
    s=s,
    marker='^',
    alpha=a,
    label='RF score=%.2f' % score_rf_multi
)

axes[1].set_xlim([-6, 6])
axes[1].set_ylim([-6, 6])
axes[1].set_xlabel('目标1')
axes[1].set_ylabel('目标2')
axes[1].set_title('图7-22 比较随机森林和多输出回归')
axes[1].legend()
axes[1].grid(alpha=0.3)


plt.suptitle('例7-12 随机森林回归实验结果', fontsize=16)

plt.tight_layout()
plt.show()