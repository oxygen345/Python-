import numpy as np


class PCA:
    def __init__(self, n_components):
        # 主成分的个数
        self.n_components = n_components

        # 具体主成分
        self.components_ = None

    def fit(self, X, eta=0.001, n_iters=1e4):
        """
        拟合 PCA
        """

        # 均值归零
        def demean(X):
            return X - np.mean(X, axis=0)

        # 方差函数
        def f(w, X):
            return np.sum(X.dot(w) ** 2) / len(X)

        # 方差函数导数
        def df(w, X):
            return X.T.dot(X.dot(w)) * 2 / len(X)

        # 将向量化简为单位向量
        def direction(w):
            return w / np.linalg.norm(w)

        # 寻找第一主成分
        def first_component(X, initial_w, eta, n_iters, epsilon=1e-8):
            w = direction(initial_w)

            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)

                last_w = w

                w = w + eta * gradient

                w = direction(w)

                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break

                cur_iter += 1

            return w

        # 均值归零
        X_pca = demean(X)

        # 初始化主成分矩阵，行为主成分，列为样本数
        self.components_ = np.empty(
            shape=(self.n_components, X.shape[1])
        )

        # 循环执行每个主成分
        for i in range(self.n_components):
            # 每次初始化一个方向向量 w
            initial_w = np.random.random(X_pca.shape[1])

            # 使用梯度上升法，得到此时 X_pca 所对应的第一主成分 w
            w = first_component(
                X_pca,
                initial_w,
                eta,
                n_iters
            )

            # 存储起来
            self.components_[i, :] = w

            # X_pca 减去样本在 w 上的所有分量，形成一个新的 X_pca
            # 以便进行下一次循环
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    # 将数据集映射到各个主成分分量中
    def transform(self, X):
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    # 将降维后的数据还原到原来的特征空间
    def inverse_transform(self, X):
        return X.dot(self.components_)
# =========================
# 测试 PCA
# =========================

np.random.seed(666)

X = np.empty((100, 2))

X[:, 0] = np.random.uniform(0, 100, size=100)
X[:, 1] = 0.75 * X[:, 0] + 3 + np.random.normal(0, 10, size=100)

pca = PCA(n_components=1)
pca.fit(X)

X_reduction = pca.transform(X)
X_restore = pca.inverse_transform(X_reduction)

print("原始数据维度：")
print(X.shape)

print("降维后数据维度：")
print(X_reduction.shape)

print("还原后数据维度：")
print(X_restore.shape)

print("主成分：")
print(pca.components_)