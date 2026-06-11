import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# =========================
# 例：Lasso L1正则化特征选择
# =========================

# 读取 Boston 房价数据集
data = fetch_openml(name='boston', version=1, as_frame=True)

# 特征数据
X = data.data

# 转成数值型，避免 category 类型报错
X = X.apply(pd.to_numeric, errors='coerce')

# 目标值
y = pd.to_numeric(data.target, errors='coerce')

# 特征名称
feature_names = X.columns

# =========================
# 标准化
# =========================

ss = StandardScaler()
X_scaled = ss.fit_transform(X)

# =========================
# Lasso 模型
# =========================

lasso = Lasso(alpha=1)

lasso.fit(X_scaled, y)

# =========================
# 输出每个特征的系数
# =========================

print('各特征的 Lasso 系数：')

for name, coef in zip(feature_names, lasso.coef_):
    print(name, format(coef, '.3f'))


# =========================
# 输出被保留下来的特征
# =========================

selected_features = feature_names[lasso.coef_ != 0]

print('\n被保留下来的特征：')
print(list(selected_features))

print('\n保留特征数量：')
print(len(selected_features))