import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 读取 Boston 房价数据
# =========================
# 如果你有 housing.data，也可以用教材原来的读取方式。
# 这里使用 fetch_openml，避免找不到 housing.data 文件。

boston = fetch_openml(
    name='boston',
    version=1,
    as_frame=True
)

df = boston.frame

# 统一列名
df.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# 转成数值型
df = df.apply(pd.to_numeric, errors='coerce')

print("房价数据前5行：")
print(df.head())


# =========================
# 2. 自定义线性回归 GD 模型
# =========================

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # w_[0] 是截距，w_[1:] 是权重
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output

            # 梯度下降更新参数
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # SSE 误差平方和
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# =========================
# 3. 取 RM 作为特征，MEDV 作为目标值
# =========================

X = df[['RM']].values
y = df['MEDV'].values


# =========================
# 4. 标准化数据
# =========================

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)

# sklearn 要求 y 转成二维后再标准化
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# =========================
# 5. 训练模型
# =========================

lr = LinearRegressionGD()

lr.fit(X_std, y_std)


# =========================
# 6. 绘制回归拟合函数
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
# 7. 绘制两个图放在一起
# =========================

plt.figure(figsize=(14, 5))

# 图7-7：SSE 收敛曲线
plt.subplot(1, 2, 1)

plt.plot(
    range(1, lr.n_iter + 1),
    lr.cost_,
    marker='o'
)

plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.title('图7-7 算法的收敛效果')
plt.grid(alpha=0.3)


# 图7-8：房间数和房价关系
plt.subplot(1, 2, 2)

lin_regplot(X_std, y_std, lr)

plt.xlabel('平均房间数 RM（标准化）')
plt.ylabel('价格为1000美元 MEDV（标准化）')
plt.title('图7-8 房间数目与房价之间的关系')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================
# 8. 预测 5 个房间的房屋价格
# =========================

num_rooms_std = sc_x.transform(np.array([[5.0]]))

price_std = lr.predict(num_rooms_std)

# 把标准化后的预测值还原成原始价格
price = sc_y.inverse_transform(price_std.reshape(-1, 1))

print('\n价格为1000美元: %.3f' % price[0][0])


# =========================
# 9. 输出斜率和截距
# =========================

print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])