import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


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

m = data.shape[0]

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)


# =========================
# 2. 代价函数
# =========================

def L_theta(theta, X_x0, y, lamb):
    """
    带 L2 正则项的代价函数
    """
    h = np.dot(X_x0, theta)

    # theta[0] 是截距项，不参与正则化
    theta_without_t0 = theta[1:]

    L = 0.5 * mean_squared_error(h, y) + 0.5 * lamb * np.sum(np.square(theta_without_t0))

    return L


# =========================
# 3. 梯度下降法
# =========================

def GD(lamb, X_x0, theta, y, alpha, T):
    for i in range(T):
        h = np.dot(X_x0, theta)

        # 截距项不参与正则化
        theta_without_t0 = np.r_[np.zeros((1, 1)), theta[1:]]

        # 梯度下降更新
        theta = theta - (
            alpha * (1 / m) * np.dot(X_x0.T, h - y)
            + lamb * theta_without_t0
        )

        if i % 50000 == 0:
            print(L_theta(theta, X_x0, y, lamb))

    return theta


# =========================
# 4. 多项式特征
# =========================

T = 1200000          # 迭代次数
degree = 11         # 多项式次数
alpha = 0.0000000006
lamb = 0.0001

theta = np.ones((degree + 1, 1))

poly_features_d = PolynomialFeatures(
    degree=degree,
    include_bias=False
)

X_poly_d = poly_features_d.fit_transform(X)

# 添加 x0 = 1
X_x0 = np.c_[np.ones((m, 1)), X_poly_d]


# =========================
# 5. 使用梯度下降训练
# =========================

print("梯度下降法训练过程中的代价函数值：")

theta_gd = GD(
    lamb=lamb,
    X_x0=X_x0,
    theta=theta,
    y=y,
    alpha=alpha,
    T=T
)

print("\n梯度下降法得到的参数 theta_gd：")
print(theta_gd)

print("\n梯度下降法最终代价函数值：")
print(L_theta(theta_gd, X_x0, y, lamb))


# =========================
# 6. 正则方程求解
# =========================

lamb2 = 10

theta2 = np.linalg.inv(
    np.dot(X_x0.T, X_x0) + lamb2 * np.identity(X_x0.shape[1])
).dot(X_x0.T).dot(y)

print("\n正则方程得到的参数 theta2：")
print(theta2)

print("\n正则方程代价函数值：")
print(L_theta(theta2, X_x0, y, lamb2))


# =========================
# 7. 使用 sklearn 的 Ridge 回归
# =========================

ridge_reg = Ridge(
    alpha=lamb2,
    solver="cholesky"
)

ridge_reg.fit(X_poly_d, y)

print("\nsklearn Ridge 截距：")
print(ridge_reg.intercept_)

print("\nsklearn Ridge 系数：")
print(ridge_reg.coef_)


def L_theta_new(intercept, coef, X, y, lamb):
    h = np.dot(X, coef.T) + intercept
    L = 0.5 * mean_squared_error(h, y) + 0.5 * lamb * np.sum(np.square(coef))
    return L


print("\nsklearn Ridge 代价函数值：")
print(
    L_theta_new(
        intercept=ridge_reg.intercept_,
        coef=ridge_reg.coef_,
        X=X_poly_d,
        y=y,
        lamb=lamb2
    )
)


# =========================
# 8. 绘图
# =========================

X_plot = np.linspace(-3, 2, 1000).reshape(-1, 1)

poly_features_d_with_bias = PolynomialFeatures(
    degree=degree,
    include_bias=True
)

X_plot_poly_bias = poly_features_d_with_bias.fit_transform(X_plot)

X_plot_poly_no_bias = poly_features_d.fit_transform(X_plot)


# 梯度下降法预测
y_plot_gd = np.dot(X_plot_poly_bias, theta_gd)

# 正则方程预测
y_plot_equation = np.dot(X_plot_poly_bias, theta2)

# sklearn Ridge 预测
y_plot_ridge = ridge_reg.predict(X_plot_poly_no_bias)


plt.figure(figsize=(18, 5))


# 图7-14：梯度下降法
plt.subplot(1, 3, 1)

plt.plot(X_plot, y_plot_gd, 'r-', label='梯度下降岭回归')
plt.plot(X, y, 'b.', label='原始数据')

plt.xlabel('x')
plt.ylabel('y')
plt.title('图7-14 岭回归的效果')
plt.legend()
plt.grid(alpha=0.3)


# 图7-15：正则方程
plt.subplot(1, 3, 2)

plt.plot(X_plot, y_plot_equation, 'r-', label='正则方程')
plt.plot(X, y, 'b.', label='原始数据')

plt.xlabel('x')
plt.ylabel('y')
plt.title('图7-15 使用正则方程求解')
plt.legend()
plt.grid(alpha=0.3)


# 图7-16：sklearn Ridge
plt.subplot(1, 3, 3)

plt.plot(X_plot, y_plot_ridge, 'r-', label='sklearn Ridge')
plt.plot(X, y, 'b.', label='原始数据')

plt.xlabel('x')
plt.ylabel('y')
plt.title('图7-16 sklearn 岭回归')
plt.legend()
plt.grid(alpha=0.3)


plt.tight_layout()
plt.show()