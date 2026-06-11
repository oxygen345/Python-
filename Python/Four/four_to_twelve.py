import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.feature_selection import VarianceThreshold


# =========================
# 例4-12 利用方差选择法进行特征选择
# =========================

# 读取 Boston 房价数据集
data = fetch_openml(name='boston', version=1, as_frame=True)

# 取特征数据
X = data.data

# 关键修改：把所有列转换成数值型
X = X.apply(pd.to_numeric, errors='coerce')

# 取目标值
y = data.target

# 查看数据维度
print('原始特征维度：')
print(X.shape)

# 查看各列标准差
print('各特征标准差：')
print(X.std())

# =========================
# 利用 sklearn 包进行方差选择
# =========================

# threshold=5 表示删除方差小于 5 的特征
vt = VarianceThreshold(threshold=5)

# 进行特征选择
xx = vt.fit_transform(X)

# 选择后维度
print('选择后的特征维度：')
print(xx.shape)