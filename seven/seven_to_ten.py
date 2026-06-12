import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 数据
# =========================

data = np.array([
    [-2.95507616, 10.94533252],
    [-0.44226119, 2.96705822],
    [-2.13294087, 6.57336839],
    [1.84990823, 5.44244467],
    [0.35139795, 2.83533936],
    [-1.77443098, 5.68004070],
    [-1.86572030, 6.34470814],
    [1.61526823, 4.77833358],
    [-2.38043687, 8.51887713],
    [-1.40513866, 4.18262786]
])

X = data[:, 0].reshape(-1, 1)
y = data[:, 1]


# =========================
# 2. 构造多项式特征
# =========================

degree = 11

poly_features_d = PolynomialFeatures(
    degree=degree,
    include_bias=False
)

X_poly_d = poly_features_d.fit_transform(X)


# =========================
# 3. ElasticNet 代价函数
# =========================
# ElasticNet = Lasso + Ridge
# r 越接近 1，越接近 Lasso
# r 越接近 0，越接近 Ridge

def L_theta_ee(intercept, coef, X, y, lamb, r):
    h = np.dot(X, coef.T) + intercept

    L_theta = (
        0.5 * mean_squared_error(h, y)
        + r * lamb * np.sum(np.abs(coef))
        + 0.5 * (1 - r) * lamb * np.sum(np.square(coef))
    )

    return L_theta


# =========================
# 4. 使用 sklearn 的 ElasticNet 回归
# =========================

elastic_net = ElasticNet(
    alpha=0.5,
    l1_ratio=0.8,
    max_iter=1000000
)

elastic_net.fit(X_poly_d, y)


# =========================
# 5. 输出参数和代价函数值
# =========================

print("ElasticNet 截距：")
print(elastic_net.intercept_)

print("\nElasticNet 系数：")
print(elastic_net.coef_)

print("\nElasticNet 代价函数值：")
print(
    L_theta_ee(
        intercept=elastic_net.intercept_,
        coef=elastic_net.coef_,
        X=X_poly_d,
        y=y,
        lamb=0.1,
        r=0.8
    )
)


# =========================
# 6. 绘制 ElasticNet 回归图像
# =========================

X_plot = np.linspace(-3, 2, 1000).reshape(-1, 1)

X_plot_poly = poly_features_d.fit_transform(X_plot)

h = np.dot(X_plot_poly, elastic_net.coef_.T) + elastic_net.intercept_

plt.figure(figsize=(8, 6))

plt.plot(
    X_plot,
    h,
    'r-',
    label='ElasticNet回归曲线'
)

plt.plot(
    X,
    y,
    'b.',
    label='原始数据'
)

plt.xlabel('x')
plt.ylabel('y')
plt.title('图7-18 弹性网络回归')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()