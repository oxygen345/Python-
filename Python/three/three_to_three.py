# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# 1. 固定随机种子，让每次生成的数据一样
np.random.seed(1)

# 2. 生成 200 个二维随机点
# randn(200, 2) 表示生成 200 行 2 列的随机数
X_xor = np.random.randn(200, 2)

# 3. 生成异或标签
# logical_xor 表示异或运算
# 如果两个条件一真一假，结果为 True
# 如果两个条件同真或同假，结果为 False
y_xor = np.logical_xor(
    X_xor[:, 0] > 0,
    X_xor[:, 1] > 0
)

# 4. 把 True 和 False 转换成类别标签
# True  转成 1
# False 转成 -1
y_xor = np.where(y_xor, 1, -1)

# 5. 绘制类别为 1 的样本点
plt.scatter(
    X_xor[y_xor == 1, 0],
    X_xor[y_xor == 1, 1],
    c='b',
    marker='x',
    label='1'
)

# 6. 绘制类别为 -1 的样本点
plt.scatter(
    X_xor[y_xor == -1, 0],
    X_xor[y_xor == -1, 1],
    c='r',
    marker='s',
    label='-1'
)

# 7. 设置坐标轴范围
plt.xlim([-3, 3])
plt.ylim([-3, 3])

# 8. 显示图例
plt.legend(loc='best')

# 9. 显示图像
plt.show()