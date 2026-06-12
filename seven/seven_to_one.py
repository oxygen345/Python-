from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 准备数据
# =========================

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]

y = [97, 86, 77, 86, 101, 86, 103, 87, 94, 78, 87, 65, 86]


# =========================
# 2. 线性回归
# =========================

slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)

print("斜率 slope =", slope)
print("截距 intercept =", intercept)
print("相关系数 rvalue =", rvalue)
print("pvalue =", pvalue)
print("标准误差 stderr =", stderr)


# =========================
# 3. 生成回归直线
# =========================

x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + intercept


# =========================
# 4. 绘图
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    y,
    color='blue',
    label='原始数据'
)

plt.plot(
    x_line,
    y_line,
    color='red',
    linewidth=2,
    label='线性回归直线'
)

plt.title('图7-2 简单线性回归')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.grid(True)

plt.show()