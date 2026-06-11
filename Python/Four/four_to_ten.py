import numpy as np
from sklearn.preprocessing import StandardScaler


# =========================
# 例4-10 利用特征标准化对数据进行缩放
# =========================

# 假设训练集包含两个特征，尺度不同
X_train = np.array([
    [100, 0.2],
    [80, 0.25],
    [70, 0.15],
    [150, 0.33],
    [200, 0.54],
    [120, 0.25],
    [135, 0.52],
    [136, 0.42],
    [210, 0.16],
    [90, 0.15]
])

# 创建 scaler 对象
scaler = StandardScaler()

# 拟合训练集
scaler.fit(X_train)

# 缩放训练集
X_train_scaled = scaler.transform(X_train)

# 构造测试集 / 验证集
X_test = np.array([
    [135, 0.25],
    [150, 0.55]
])

# 缩放测试集 / 验证集
X_test_scaled = scaler.transform(X_test)

# 查看结果
print('拟合训练集：')
print(X_train)

print('缩放训练集：')
print(X_train_scaled)

print('缩放验证集：')
print(X_test_scaled)