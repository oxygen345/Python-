import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


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
# 3. 定义 Lasso 代价函数
# =========================

def L_theta_new(intercept, coef, X, y, lamb):
    """
    Lasso 回归的代价函数：
    均方误差 + L1 正则项
    """
    h = np.dot(X, coef.T) + intercept

    L_theta = (
        0.5 * mean_squared_error(h, y)
        + 0.5 * lamb * np.sum(np.abs(coef))
    )

    return L_theta


# =========================
# 4. 使用 sklearn 的 Lasso 回归
# =========================

lamb = 0.025

lasso_reg = Lasso(
    alpha=lamb,
    max_iter=1000000
)

lasso_reg.fit(X_poly_d, y)


# =========================
# 5. 输出参数和代价函数值
# =========================

print("Lasso 截距：")
print(lasso_reg.intercept_)

print("\nLasso 系数：")
print(lasso_reg.coef_)

print("\nLasso 代价函数值：")
print(
    L_theta_new(
        intercept=lasso_reg.intercept_,
        coef=lasso_reg.coef_,
        X=X_poly_d,
        y=y,
        lamb=lamb
    )
)


# =========================
# 6. 绘制 Lasso 回归图像
# =========================

X_plot = np.linspace(-3, 2, 1000).reshape(-1, 1)

X_plot_poly = poly_features_d.fit_transform(X_plot)

h = np.dot(X_plot_poly, lasso_reg.coef_.T) + lasso_reg.intercept_

plt.figure(figsize=(8, 6))

plt.plot(
    X_plot,
    h,
    'r-',
    label='Lasso回归曲线'
)

plt.plot(
    X,
    y,
    'b.',
    label='原始数据'
)

plt.xlabel('x')
plt.ylabel('y')
plt.title('图7-17 Lasso回归')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()