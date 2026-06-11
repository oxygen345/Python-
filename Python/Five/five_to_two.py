import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

# 设置中文显示和负号显示
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']


# =========================
# LDA 类
# =========================

class LDA:
    def __init__(self):
        self.w = None
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    # 计算协方差矩阵
    def calculate_covariance_matrix(self, X, Y=None):
        m = X.shape[0]

        X = X - np.mean(X, axis=0)

        if Y is None:
            Y = X
        else:
            Y = Y - np.mean(Y, axis=0)

        return 1 / m * np.matmul(X.T, Y)

    # 数据转换
    def transform(self, X, y):
        self.fit(X, y)
        X_transform = X.dot(self.w)
        return X_transform

    # LDA 拟合过程
    def fit(self, X, y):
        # 按类别划分
        X0 = X[y.reshape(-1) == 0]
        X1 = X[y.reshape(-1) == 1]

        # 计算两类数据变量的协方差矩阵
        sigma0 = self.calculate_covariance_matrix(X0)
        sigma1 = self.calculate_covariance_matrix(X1)

        # 计算类内散度矩阵
        Sw = sigma0 + sigma1

        # 分别计算两类数据的均值
        u0 = X0.mean(axis=0)
        u1 = X1.mean(axis=0)

        # 均值差
        mean_diff = np.atleast_1d(u0 - u1)

        # 对类内矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)

        # 计算类内散度矩阵的逆
        Sw_inv = np.dot(
            np.dot(V.T, np.linalg.pinv(np.diag(S))),
            U.T
        )

        # 计算投影向量 w
        self.w = Sw_inv.dot(mean_diff)

        return self.w

    # LDA 分类预测
    def predict(self, X):
        y_pred = []

        for sample in X:
            h = sample.dot(self.w)

            if h < 0:
                y = 1
            else:
                y = 0

            y_pred.append(y)

        return np.array(y_pred)

    # 生成训练数据
    def get_train_data(self, data_size=100):
        data_label = np.zeros((2 * data_size, 1))

        # 类别0
        x1 = np.reshape(
            np.random.normal(1, 0.6, data_size),
            (data_size, 1)
        )

        y1 = np.reshape(
            np.random.normal(1, 0.8, data_size),
            (data_size, 1)
        )

        data_train = np.concatenate((x1, y1), axis=1)
        data_label[0:data_size, :] = 0

        # 类别1
        x2 = np.reshape(
            np.random.normal(-1, 0.3, data_size),
            (data_size, 1)
        )

        y2 = np.reshape(
            np.random.normal(-1, 0.5, data_size),
            (data_size, 1)
        )

        data_train = np.concatenate(
            (
                data_train,
                np.concatenate((x2, y2), axis=1)
            ),
            axis=0
        )

        data_label[data_size:2 * data_size, :] = 1

        self.train_data = data_train
        self.train_label = data_label

        return data_train, data_label

    # 生成测试数据
    def get_test_data(self, data_size=10):
        testdata_label = np.zeros((2 * data_size, 1))

        # 类别0
        x1 = np.reshape(
            np.random.normal(1, 0.6, data_size),
            (data_size, 1)
        )

        y1 = np.reshape(
            np.random.normal(1, 0.8, data_size),
            (data_size, 1)
        )

        data_test = np.concatenate((x1, y1), axis=1)
        testdata_label[0:data_size, :] = 0

        # 类别1
        x2 = np.reshape(
            np.random.normal(-1, 0.3, data_size),
            (data_size, 1)
        )

        y2 = np.reshape(
            np.random.normal(-1, 0.5, data_size),
            (data_size, 1)
        )

        data_test = np.concatenate(
            (
                data_test,
                np.concatenate((x2, y2), axis=1)
            ),
            axis=0
        )

        testdata_label[data_size:2 * data_size, :] = 1

        self.test_data = data_test
        self.test_label = testdata_label

        return data_test, testdata_label

    # 二维图和三维图放在一个窗口里
    def plot_all_decision(self):
        train_data = self.train_data
        train_label = self.train_label
        test_data = self.test_data
        test_label = self.test_label
        w = self.w

        fig = plt.figure(figsize=(14, 6))

        # =========================
        # 左图：二维分类界面
        # =========================
        ax1 = fig.add_subplot(1, 2, 1)

        x = np.arange(-2, 2.1, 0.1)

        # 二维分类边界
        y = w[0] * x / w[1]

        # 类别0：蓝色
        ax1.scatter(
            train_data[:100, 0],
            train_data[:100, 1],
            c='dodgerblue',
            marker='+',
            s=70,
            label='类别0'
        )

        # 类别1：橙色
        ax1.scatter(
            train_data[100:, 0],
            train_data[100:, 1],
            c='orange',
            marker='o',
            s=45,
            label='类别1'
        )

        # 测试数据：红色方块
        ax1.scatter(
            test_data[:, 0],
            test_data[:, 1],
            c='red',
            marker='s',
            s=65,
            label='测试数据'
        )

        # 分类边界：紫色虚线
        ax1.plot(
            x,
            y,
            color='purple',
            linestyle='--',
            linewidth=2,
            label='二维分类界面'
        )

        ax1.set_title('图5-9 二分类界面')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # =========================
        # 右图：三维分类界面
        # =========================
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # 类别0：蓝色
        ax2.scatter(
            train_data[:100, 0],
            train_data[:100, 1],
            train_label[:100, 0],
            c='dodgerblue',
            marker='+',
            s=60,
            label='类别0'
        )

        # 类别1：橙色
        ax2.scatter(
            train_data[100:, 0],
            train_data[100:, 1],
            train_label[100:, 0],
            c='orange',
            marker='o',
            s=45,
            label='类别1'
        )

        # 测试数据：红色方块
        ax2.scatter(
            test_data[:, 0],
            test_data[:, 1],
            test_label[:, 0],
            c='red',
            marker='s',
            s=60,
            label='测试数据'
        )

        # 三维分类平面
        x1 = np.arange(-2, 2.1, 0.1)
        x2 = np.arange(-3, 3.1, 0.1)

        x1, x2 = np.meshgrid(x1, x2)

        y_surface = w[0] * x1 + w[1] * x2

        ax2.plot_surface(
            x1,
            x2,
            y_surface,
            rstride=1,
            cstride=1,
            cmap=plt.cm.coolwarm,
            alpha=0.45
        )

        ax2.set_title('图5-10 LDA三维分类界面')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('类别 / 投影值')
        ax2.legend()

        plt.suptitle('LDA分类界面与三维投影结果', fontsize=16)
        plt.tight_layout()
        plt.show()


# =========================
# 主程序
# =========================

if __name__ == "__main__":
    # 固定随机种子，保证每次运行结果一致
    np.random.seed(666)

    # 创建 LDA 对象
    lda = LDA()

    # 产生训练数据和测试数据
    train_data, train_label = lda.get_train_data(data_size=100)
    test_data, test_label = lda.get_test_data(data_size=10)

    print("训练数据：", train_data.shape)
    print("训练标签：", train_label.shape)
    print("测试数据：", test_data.shape)
    print("测试标签：", test_label.shape)

    # 训练 LDA
    w = lda.fit(train_data, train_label)

    print("分界面权向量 w：")
    print(w)

    # 预测测试集
    y_pred = lda.predict(test_data)

    print("测试集预测值为：")
    print(y_pred)

    print("测试集真实值为：")
    print(test_label.reshape(-1).astype(int))

    # 计算准确率
    acc = np.sum(y_pred == test_label.reshape(-1)) / len(y_pred)

    print("测试集预测精度为 acc=")
    print(acc)

    # 二维图和三维图放在同一个窗口
    lda.plot_all_decision()