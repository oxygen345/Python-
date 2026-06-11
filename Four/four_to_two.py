import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 利用 3σ 原则分析异常值
# =========================

# 生成随机数据
data = pd.Series(np.random.randn(10000) * 100)

# 计算均值和标准差
u = data.mean()
std = data.std()

print('均值为:%.3f, 标准差为:%.3f' % (u, std))

# 创建画布
fig = plt.figure(figsize=(10, 6))

# 第一张图：密度曲线
ax1 = fig.add_subplot(2, 1, 1)

data.plot(
    kind='kde',
    grid=True,
    style='--k',
    title='密度曲线',
    ax=ax1
)

# 第二张图：异常值散点图
ax2 = fig.add_subplot(2, 1, 2)

# 根据 3σ 原则筛选异常值
error = data[np.abs(data - u) > 3 * std]

# 剔除异常值后的正常数据
data_c = data[np.abs(data - u) <= 3 * std]

print('异常值共%i条' % len(error))

# 正常数据
ax2.scatter(
    data_c.index,
    data_c,
    color='k',
    marker='.',
    alpha=0.3
)

# 异常值
ax2.scatter(
    error.index,
    error,
    color='r',
    marker='.',
    alpha=0.5
)

ax2.set_xlim([-10, 10010])
ax2.grid()

plt.show()