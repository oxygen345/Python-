import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# =========================
# 设置中文显示
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 加载并处理数据
# =========================

iris = load_iris()

# 只取后100个样本，做二分类
# 只取第2、3个特征，方便画二维决策区域
X = iris.data[50:, [1, 2]]
y = iris.target[50:]

# 标签编码，把类别变成0和1
le = LabelEncoder()
y = le.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=1,
    stratify=y
)


# =========================
# 2. 构建三个分类器
# =========================

# 新版 sklearn 中不需要写 penalty='l2'
clf1 = LogisticRegression(
    C=0.001,
    random_state=1,
    max_iter=1000
)

clf2 = DecisionTreeClassifier(
    max_depth=1,
    criterion='entropy',
    random_state=0
)

clf3 = KNeighborsClassifier(
    n_neighbors=1,
    p=2,
    metric='minkowski'
)


# =========================
# 3. 管道标准化
# =========================

pipe1 = Pipeline([
    ['sc', StandardScaler()],
    ['clf', clf1]
])

pipe3 = Pipeline([
    ['sc', StandardScaler()],
    ['clf', clf3]
])


# =========================
# 4. 10折交叉验证评价单个模型
# =========================

clf_labels = ['逻辑回归', '决策树', 'KNN']

print('10折交叉验证：\n')

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(
        estimator=clf,
        X=X_train,
        y=y_train,
        cv=10,
        scoring='roc_auc'
    )

    print(
        'ROC AUC: %.2f (+/- %.2f) [%s]'
        % (scores.mean(), scores.std(), label)
    )


# =========================
# 5. 多投票机制组合分类器
# =========================

mv_clf = VotingClassifier(
    estimators=[
        ('pipe1', pipe1),
        ('clf2', clf2),
        ('pipe3', pipe3)
    ],
    voting='soft'
)

clf_labels += ['多投票机制组合分类器']

all_clf = [pipe1, clf2, pipe3, mv_clf]

print('\n加入多投票机制组合分类器后：\n')

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(
        estimator=clf,
        X=X_train,
        y=y_train,
        cv=10,
        scoring='roc_auc'
    )

    print(
        'ROC AUC: %.2f (+/- %.2f) [%s]'
        % (scores.mean(), scores.std(), label)
    )


# =========================
# 6. 绘制 ROC 曲线
# =========================

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

plt.figure(figsize=(8, 6))

for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # 训练模型并计算预测概率
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(
        y_true=y_test,
        y_score=y_pred
    )

    # 计算 AUC
    roc_auc = auc(
        x=fpr,
        y=tpr
    )

    plt.plot(
        fpr,
        tpr,
        color=clr,
        linestyle=ls,
        linewidth=2,
        label='%s (auc=%0.2f)' % (label, roc_auc)
    )

# 随机猜测参考线
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    color='gray',
    linewidth=2
)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('假阳性率(FPR)')
plt.ylabel('真阳性率(TPR)')
plt.title('图6-6 各方法分类效果')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# =========================
# 7. 绘制分类器决策区域
# =========================

# 重新标准化训练数据，方便画决策区域
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

# 决策区域范围
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1

y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    sharex='col',
    sharey='row',
    figsize=(8, 6)
)

# 注意：这里的模型都用标准化后的数据训练
clf1_std = LogisticRegression(
    C=0.001,
    random_state=1,
    max_iter=1000
)

clf2_std = DecisionTreeClassifier(
    max_depth=1,
    criterion='entropy',
    random_state=0
)

clf3_std = KNeighborsClassifier(
    n_neighbors=1,
    p=2,
    metric='minkowski'
)

mv_clf_std = VotingClassifier(
    estimators=[
        ('lr', clf1_std),
        ('dt', clf2_std),
        ('knn', clf3_std)
    ],
    voting='soft'
)

decision_clfs = [
    clf1_std,
    clf2_std,
    clf3_std,
    mv_clf_std
]

decision_titles = [
    '逻辑回归',
    '决策树',
    'KNN',
    '多投票机制组合分类器'
]

for idx, clf, title in zip(
    product([0, 1], [0, 1]),
    decision_clfs,
    decision_titles
):
    clf.fit(X_train_std, y_train)

    Z = clf.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    axes[idx[0], idx[1]].contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap=plt.cm.RdYlBu
    )

    # 类别0
    axes[idx[0], idx[1]].scatter(
        X_train_std[y_train == 0, 0],
        X_train_std[y_train == 0, 1],
        c='blue',
        marker='^',
        s=50,
        edgecolor='black',
        label='类别0'
    )

    # 类别1
    axes[idx[0], idx[1]].scatter(
        X_train_std[y_train == 1, 0],
        X_train_std[y_train == 1, 1],
        c='red',
        marker='o',
        s=50,
        edgecolor='black',
        label='类别1'
    )

    axes[idx[0], idx[1]].set_title(title)
    axes[idx[0], idx[1]].grid(alpha=0.3)

# 公共坐标轴文字
fig.text(
    0.5,
    0.04,
    '萼片宽度[标准化]',
    ha='center',
    fontsize=12
)

fig.text(
    0.04,
    0.5,
    '花片长度[标准化]',
    va='center',
    rotation=90,
    fontsize=12
)

plt.suptitle('图6-7 决策区域', fontsize=16)
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.show()