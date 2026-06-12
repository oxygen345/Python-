import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 读取 Boston 房价数据集
# =========================

boston = fetch_openml(
    name='boston',
    version=1,
    as_frame=True
)

df = boston.frame

df.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df = df.apply(pd.to_numeric, errors='coerce')

print("房价数据前5行：")
print(df.head())


# =========================
# 2. 取 RM 作为特征，MEDV 作为目标值
# =========================
# RM：平均房间数
# MEDV：房价，单位为 1000 美元

X = df[['RM']].values
y = df['MEDV'].values


# =========================
# 3. 使用 sklearn 建立线性回归模型
# =========================

slr = LinearRegression()

slr.fit(X, y)

y_pred = slr.predict(X)


# =========================
# 4. 输出斜率和截距
# =========================

print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# =========================
# 5. 定义绘图函数
# =========================

def lin_regplot(X, y, model):
    plt.scatter(
        X,
        y,
        c='steelblue',
        edgecolor='white',
        s=70,
        label='训练样本'
    )

    plt.plot(
        X,
        model.predict(X),
        color='black',
        lw=2,
        label='回归直线'
    )

    plt.legend()


# =========================
# 6. 绘制图7-9
# =========================

plt.figure(figsize=(8, 6))

lin_regplot(X, y, slr)

plt.xlabel('平均房间数 RM')
plt.ylabel('价格1000美元 MEDV')
plt.title('图7-9 MEDV与RM的关系与梯度下降法比较')

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()