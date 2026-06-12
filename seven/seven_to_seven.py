# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, RANSACRegressor


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 读取 Boston 房价数据集
# =========================

base_dir = Path(__file__).resolve().parent
housing_path = base_dir / "housing.data"

if not housing_path.exists():
    raise FileNotFoundError(
        f"没有找到数据集文件：{housing_path}\n"
        "请把 housing.data 放到 seven_to_seven.py 同一个文件夹里。"
    )

df = pd.read_csv(
    housing_path,
    header=None,
    sep=r"\s+"
)

df.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

print("Boston 房价数据前5行：")
print(df.head())


# =========================
# 2. 取 RM 作为特征，MEDV 作为目标值
# =========================

X = df[['RM']].values
y = df['MEDV'].values


# =========================
# 3. 使用 RANSAC 拟合鲁棒线性回归模型
# =========================
# 新版 sklearn 已经不再使用 residual_metric
# 用 loss='absolute_error' 代替

ransac = RANSACRegressor(
    estimator=LinearRegression(),
    max_trials=100,
    min_samples=50,
    loss='absolute_error',
    residual_threshold=5.0,
    random_state=0
)

ransac.fit(X, y)


# =========================
# 4. 获取内点和异常点
# =========================

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))


# =========================
# 5. 绘图
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[inlier_mask].ravel(),
    y[inlier_mask],
    c='blue',
    marker='o',
    label='Inliers'
)

plt.scatter(
    X[outlier_mask].ravel(),
    y[outlier_mask],
    c='lightgreen',
    marker='s',
    label='Outliers'
)

plt.plot(
    line_X,
    line_y_ransac,
    color='red',
    linewidth=2,
    label='RANSAC回归线'
)

plt.xlabel('平均房间数 RM')
plt.ylabel('价格1000美元 MEDV')
plt.title('图7-10 点与线性拟合效果')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# =========================
# 6. 输出模型斜率和截距
# =========================

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)