import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 例4-3 箱形图分析
# =========================

data = pd.Series(np.random.randn(10000) * 100)

fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_subplot(2, 1, 1)

color = dict(
    boxes='DarkGreen',
    whiskers='DarkOrange',
    medians='DarkBlue',
    caps='Gray'
)

data.plot.box(
    vert=False,
    grid=True,
    color=color,
    ax=ax1,
    label='样本数据'
)

# 从箱形图观察数据分布情况，以内限为界
s = data.describe()
print('数据分布情况：', s)

# 基本统计量
q1 = s['25%']
q3 = s['75%']

iqr = q3 - q1

mi = q1 - 1.5 * iqr
ma = q3 + 1.5 * iqr

print('分位差为:%.3f，下限为:%.3f，上限为:%.3f' % (iqr, mi, ma))

# 计算分位差
ax2 = fig.add_subplot(2, 1, 2)

error = data[(data < mi) | (data > ma)]

data_c = data[(data >= mi) & (data <= ma)]

# 筛选出异常值 error，剔除异常值之后的数据 data_c
print('异常值共%i条' % len(error))

# 图表表达
plt.scatter(
    data_c.index,
    data_c,
    color='k',
    marker='.',
    alpha=0.3
)

plt.scatter(
    error.index,
    error,
    color='r',
    marker='.',
    alpha=0.5
)

plt.xlim([-10, 10010])
plt.grid()

plt.show()