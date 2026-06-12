import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 设置中文显示
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 构建披萨数据集
# =========================

pizza = pd.DataFrame({
    'Diameter': [6, 8, 10, 14, 18, 8, 9, 11, 16, 12],
    'Toppings': [2, 1, 0, 2, 0, 2, 0, 2, 2, 0],
    'Price': [7, 9, 13, 17.5, 18, 11, 8.5, 15, 18, 11]
}, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

pizza.index.name = 'Id'

print('披萨数据：')
print(pizza)


# =========================
# 2. 前5行作为训练集，后5行作为测试集
# =========================

X = pizza.iloc[:5, :2].values
y = pizza.iloc[:5, 2].values.reshape(-1, 1)

print('\n训练集 X：')
print(X)

print('\n训练集 y：')
print(y)


# =========================
# 3. 给 X 添加一列 1，用于计算截距 b
# =========================

ones = np.ones(X.shape[0]).reshape(-1, 1)

X_b = np.hstack((X, ones))

print('\n添加截距列后的 X_b：')
print(X_b)


# =========================
# 4. 使用正规方程求参数
# w = (X^T X)^(-1) X^T y
# =========================

w_all = np.dot(
    np.dot(
        np.linalg.inv(
            np.dot(X_b.T, X_b)
        ),
        X_b.T
    ),
    y
)

print('\n完整参数 w_all：')
print(w_all)


# =========================
# 5. 分离权重 w 和截距 b
# =========================

w = w_all[:-1]
b = w_all[-1]

print('\n权重 w：')
print(w)

print('\n截距 b：')
print(b)


# =========================
# 6. 输出回归方程
# =========================

print('\n回归方程为：')
print(
    'Price = %.8f * Diameter + %.8f * Toppings + %.4f'
    % (w[0, 0], w[1, 0], b[0])
)


# =========================
# 7. 使用后5行数据进行预测
# =========================

X_test = pizza.iloc[-5:, :2].values
y_test = pizza.iloc[-5:, 2].values.reshape(-1, 1)

print('\n测试集 X_test：')
print(X_test)

print('\n目标值 y_test：')
print(y_test)

y_pred = np.dot(X_test, w) + b

print('\n预测值 y_pred：')
print(y_pred)


# =========================
# 8. 真实值和预测值对比
# =========================

result = pd.DataFrame({
    'Diameter': X_test[:, 0],
    'Toppings': X_test[:, 1],
    '真实价格': y_test.ravel(),
    '预测价格': y_pred.ravel()
})

print('\n真实值与预测值对比：')
print(result)


# =========================
# 9. 绘制真实值和预测值对比图
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    y_test,
    marker='o',
    color='blue',
    label='真实值'
)

plt.plot(
    y_pred,
    marker='s',
    color='red',
    linestyle='--',
    label='预测值'
)

plt.title('图7-3 多元线性回归预测披萨价格')
plt.xlabel('测试样本编号')
plt.ylabel('披萨价格')
plt.legend()
plt.grid(alpha=0.3)

plt.show()