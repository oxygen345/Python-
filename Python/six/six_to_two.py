import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


# =========================
# 设置中文显示
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 构建实验数据集
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
# 2. 硬投票和软投票效果对比
# =========================

log_clf = LogisticRegression(
    random_state=520,
    max_iter=1000
)

rnd_clf = RandomForestClassifier(
    random_state=520
)

# 新版 sklearn 推荐用 CalibratedClassifierCV 代替 SVC(probability=True)
svm_clf = CalibratedClassifierCV(
    SVC(random_state=520),
    cv=3
)


# ---------- 硬投票 ----------
voting_clf_hard = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rnd_clf),
        ('svc', svm_clf)
    ],
    voting='hard'
)

print('硬投票效果：')

for clf in (log_clf, rnd_clf, svm_clf, voting_clf_hard):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(
        clf.__class__.__name__,
        round(accuracy_score(y_test, y_pred), 3)
    )


# ---------- 软投票 ----------
log_clf = LogisticRegression(
    random_state=520,
    max_iter=1000
)

rnd_clf = RandomForestClassifier(
    random_state=520
)

svm_clf = CalibratedClassifierCV(
    SVC(random_state=520),
    cv=3
)

voting_clf_soft = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rnd_clf),
        ('svc', svm_clf)
    ],
    voting='soft'
)

print('\n软投票效果：')

for clf in (log_clf, rnd_clf, svm_clf, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(
        clf.__class__.__name__,
        round(accuracy_score(y_test, y_pred), 3)
    )


# =========================
# 3. Bagging策略效果
# =========================

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print('\n用Bagging策略效果：')
print(round(accuracy_score(y_test, y_pred), 3))


# =========================
# 4. 不用Bagging策略效果
# =========================

tree_clf = DecisionTreeClassifier(
    random_state=42
)

tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

print('\n不用Bagging策略效果：')
print(round(accuracy_score(y_test, y_pred_tree), 3))


# =========================
# 5. 绘制决策边界函数
# =========================

def plot_decision_boundary(ax, clf, X, y, axes=(-1.5, 2.5, -1.0, 1.5), contour=True):
    x1s = np.linspace(axes[0], axes[1], 200)
    x2s = np.linspace(axes[2], axes[3], 200)

    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]

    y_pred = clf.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap([
        '#FFF2B2',
        '#B7C9FF'
    ])

    ax.contourf(
        x1,
        x2,
        y_pred,
        cmap=custom_cmap,
        alpha=0.5
    )

    if contour:
        custom_cmap2 = ListedColormap([
            '#B8860B',
            '#4169E1'
        ])

        ax.contour(
            x1,
            x2,
            y_pred,
            cmap=custom_cmap2,
            alpha=0.8
        )

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color='orange',
        marker='o',
        alpha=0.7,
        label='类别0'
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='blue',
        marker='s',
        alpha=0.7,
        label='类别1'
    )

    ax.axis(axes)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    ax.grid(alpha=0.3)


# =========================
# 6. 三张图放在同一个窗口
# =========================

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(18, 5)
)

# 左图：原始散点图
axes[0].scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='orange',
    marker='o',
    alpha=0.7,
    label='类别0'
)

axes[0].scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='blue',
    marker='s',
    alpha=0.7,
    label='类别1'
)

axes[0].set_title('图6-8 构建的散点图')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].legend()
axes[0].grid(alpha=0.3)


# 中图：单棵决策树
plot_decision_boundary(
    axes[1],
    tree_clf,
    X,
    y
)

axes[1].set_title('决策树')


# 右图：Bagging
plot_decision_boundary(
    axes[2],
    bag_clf,
    X,
    y
)

axes[2].set_title('决策树与Bagging')


plt.suptitle('图6-8 与 图6-9 Bagging算法效果展示', fontsize=16)
plt.tight_layout()
plt.show()


# =========================
# 7. OOB袋外数据的作用
# =========================

bag_clf_oob = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    oob_score=True
)

bag_clf_oob.fit(X_train, y_train)

print('\nOOB袋外效果：')
print(bag_clf_oob.oob_score_)

y_pred_oob = bag_clf_oob.predict(X_test)

print('\n测试集计算结果：')
print(round(accuracy_score(y_test, y_pred_oob), 3))