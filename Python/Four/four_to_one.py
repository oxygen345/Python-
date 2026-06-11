import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


# =========================
# 1. 简单缺失值填补
# =========================

s = pd.Series([12, np.nan, 33, 45, np.nan, 66, 67, 80, 90, np.nan])

u = s.mean()
me = s.median()
mod = s.mode()

print('均值为: %.2f, 中位数为: %.2f' % (u, me))
print('众数为:', mod.tolist())

# 用均值填补
s1 = s.copy()
s1.fillna(u, inplace=True)
print('均值填补:')
print(s1)

# 用中位数填补
s2 = s.copy()
s2.fillna(me, inplace=True)
print('中位数填补:')
print(s2)

# 用众数填补
s3 = s.copy()
s3.fillna(mod[0], inplace=True)
print('众数填补:')
print(s3)

# 临近值填补，用前值插补
s4 = s.copy()
s4.ffill(inplace=True)
print('临近值填补:')
print(s4)


# =========================
# 2. 拉格朗日插值法
# =========================

data = pd.Series(np.random.randn(100) * 100)

# 设置缺失值
data[[3, 6, 33, 56, 45, 66, 67, 80, 90]] = np.nan

print('拉格朗日插值法:')
print(data.head())

print('总数据量:%i' % len(data))

data_na = data[data.isnull()]

print('缺失值数据量:%i' % len(data_na))
print('缺失数据占比:%.2f%%' % (len(data_na) / len(data) * 100))

# 中位数填充
data_c = data.fillna(data.median())


# 定义拉格朗日插值函数
def na_c(s, n, k=5):
    start = max(n - k, 0)
    end = min(n + k + 1, len(s))

    y = s.iloc[start:end]
    y = y[y.notnull()]

    return lagrange(y.index, list(y))(n)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 原始数据分布
data.plot.box(
    ax=axes[0],
    grid=True,
    title='数据分布'
)

# 删除缺失值后的分布
data.dropna().plot(
    kind='kde',
    style='--r',
    ax=axes[1],
    grid=True,
    title='删除缺失值',
    xlim=[-50, 150]
)

# 中位数填充后的分布
data_c.plot(
    kind='kde',
    style='--b',
    ax=axes[2],
    grid=True,
    title='缺失值填充中位数',
    xlim=[-50, 150]
)

# 拉格朗日插值
na_re = []

for i in range(len(data)):
    if pd.isnull(data.iloc[i]):
        value = na_c(data, i)
        data.iloc[i] = value
        print(value)
        na_re.append(value)

# 清除插值后仍存在的缺失值
data.dropna(inplace=True)

# 拉格朗日插值后的分布
data.plot(
    kind='kde',
    style='--k',
    ax=axes[3],
    grid=True,
    title='拉格朗日插值后',
    xlim=[-50, 150]
)

plt.show()