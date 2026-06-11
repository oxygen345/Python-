import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from scipy import stats


# =========================
# 例4-13 利用相关系数法进行特征选择
# =========================

# 读取 Boston 房价数据集
data = fetch_openml(name='boston', version=1, as_frame=True)

# 获取特征数据
X = data.data

# 把所有特征转成数值型，避免 category 类型报错
X = X.apply(pd.to_numeric, errors='coerce')

# 获取目标值
y = data.target

# 把目标值也转成数值型
y = pd.to_numeric(y, errors='coerce')

# 转成 numpy 数组
X_values = X.values
y_values = y.values

# 保存每个特征的 Pearson 相关系数和 P 值
pearson_result = []

# 逐个特征计算与目标值 y 的相关系数
for i in range(X_values.shape[1]):
    X_pear = stats.pearsonr(X_values[:, i], y_values)
    pearson_result.append(X_pear)
    print(X_pear)