import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 例5-1 手写数字降噪实例
# =========================

# 1. 加载 digits 手写数字数据集
digits = datasets.load_digits()

X = digits.data
y = digits.target

# 添加正态分布噪声
np.random.seed(666)
noisy_digits = X + np.random.normal(0, 4, size=X.shape)

# 每个数字取10张，共100张
example_digits = noisy_digits[y == 0, :][:10]

for num in range(1, 10):
    example_digits = np.vstack([
        example_digits,
        noisy_digits[y == num, :][:10]
    ])

print('带噪声数字数据形状：')
print(example_digits.shape)


# =========================
# 2. 使用 PCA 进行降噪
# =========================

pca = PCA(0.5).fit(noisy_digits)

print('PCA保留的主成分个数：')
print(pca.n_components_)

components = pca.transform(example_digits)

filtered_digits = pca.inverse_transform(components)


# =========================
# 3. 同时展示两张图片
# =========================

def plot_digits_compare(noisy_data, filtered_data):
    fig, axes = plt.subplots(
        10,
        20,
        figsize=(16, 8),
        subplot_kw={
            'xticks': [],
            'yticks': []
        },
        gridspec_kw={
            'hspace': 0.1,
            'wspace': 0.1
        }
    )

    # 左边10列：带噪声的数字
    for i in range(100):
        row = i // 10
        col = i % 10

        axes[row, col].imshow(
            noisy_data[i].reshape(8, 8),
            cmap='binary',
            interpolation='nearest',
            clim=(0, 16)
        )

    # 右边10列：PCA降噪后的数字
    for i in range(100):
        row = i // 10
        col = i % 10 + 10

        axes[row, col].imshow(
            filtered_data[i].reshape(8, 8),
            cmap='binary',
            interpolation='nearest',
            clim=(0, 16)
        )

    # 添加标题
    fig.text(0.25, 0.95, '带噪声的数字', ha='center', fontsize=16)
    fig.text(0.75, 0.95, 'PCA降噪后的数字', ha='center', fontsize=16)

    plt.suptitle('图5-1 PCA手写数字降噪效果对比', fontsize=18)
    plt.show()


plot_digits_compare(example_digits, filtered_digits)