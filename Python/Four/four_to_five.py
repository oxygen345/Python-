# 导入第三方包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 随机生成两组二维正态分布数据
# =========================

np.random.seed(1234)

mean1 = [0.5, 0.5]
cov1 = [[0.3, 0], [0, 0.1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 5000).T

mean2 = [0, 8]
cov2 = [[0.8, 0], [0, 2]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 5000).T


# =========================
# 2. 将两组数据合并到数据框中
# =========================

X = pd.DataFrame(
    np.concatenate(
        [
            np.array([x1, y1]),
            np.array([x2, y2])
        ],
        axis=1
    ).T
)

X.rename(columns={0: 'x1', 1: 'x2'}, inplace=True)


# =========================
# 3. KMeans 离群点检测函数
# =========================

def kmeans_outliers(data, clusters, is_scale=True):
    kmeans = KMeans(n_clusters=clusters, random_state=1234)
    cluster_res = []

    if is_scale:
        std_data = scale(data)
        kmeans.fit(std_data)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        for label in set(labels):
            diff = std_data[np.array(labels) == label, ] - np.array(centers[label])
            dist = np.sum(np.square(diff), axis=1)

            UL = dist.mean() + 3 * dist.std()
            OutLine = np.where(dist > UL, 1, 0)

            raw_data = data.loc[np.array(labels) == label, ]

            new_data = pd.DataFrame({
                'Label': label,
                'Dist': dist,
                'Outlier': OutLine
            })

            raw_data.index = new_data.index = range(raw_data.shape[0])

            cluster_res.append(pd.concat([raw_data, new_data], axis=1))

    else:
        kmeans.fit(data)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        for label in set(labels):
            diff = np.array(data.loc[np.array(labels) == label, ]) - np.array(centers[label])
            dist = np.sum(np.square(diff), axis=1)

            UL = dist.mean() + 3 * dist.std()
            OutLine = np.where(dist > UL, 1, 0)

            raw_data = data.loc[np.array(labels) == label, ]

            new_data = pd.DataFrame({
                'Label': label,
                'Dist': dist,
                'Outlier': OutLine
            })

            raw_data.index = new_data.index = range(raw_data.shape[0])

            cluster_res.append(pd.concat([raw_data, new_data], axis=1))

    return pd.concat(cluster_res)


# =========================
# 4. 调用函数检测异常值
# =========================

res = kmeans_outliers(X, 2, False)

print(res.head())
print('异常值数量：')
print(res['Outlier'].value_counts())


# =========================
# 5. 两个图并列显示，并用不同颜色区分异常点
# =========================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：随机分布数据
axes[0].scatter(
    x1,
    y1,
    color='gray',
    s=10,
    alpha=0.6
)

axes[0].scatter(
    x2,
    y2,
    color='gray',
    s=10,
    alpha=0.6
)

axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('(a) 随机分布数据')


# 右图：异常检测结果
normal_data = res[res['Outlier'] == 0]
outlier_data = res[res['Outlier'] == 1]

# 正常点：蓝色
axes[1].scatter(
    normal_data['x1'],
    normal_data['x2'],
    color='blue',
    s=10,
    alpha=0.6,
    label='0'
)

# 异常点：红色
axes[1].scatter(
    outlier_data['x1'],
    outlier_data['x2'],
    color='red',
    s=12,
    alpha=0.8,
    label='1'
)

axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title('(b) 异常检测结果')
axes[1].legend(loc='best')

plt.suptitle('图4-6 K均值异常检测处理')
plt.tight_layout()
plt.show()