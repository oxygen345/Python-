import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn import linear_model
from sklearn.metrics import r2_score

import statsmodels.api as sm


# =========================
# 1. 读取 Boston 数据集
# =========================

boston = fetch_openml(
    name='boston',
    version=1,
    as_frame=True
)

x = boston.data
y = boston.target

# 转成数值型，避免 category 类型报错
x = x.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# 目标变量改成 DataFrame，列名保持和教材类似
y_df = pd.DataFrame(y, columns=['MEDV'])

print('Boston 房价数据前5行：')
print(x.head())

print('\n目标变量前5行：')
print(y_df.head())


# =========================
# 2. 使用 sklearn 进行线性回归
# =========================

method = linear_model.LinearRegression()

getmodel = method.fit(x, y)

py = getmodel.predict(x)

r_square = r2_score(y, py)

print('\n使用 sklearn 线性回归：')
print('R平方：{:.2f}'.format(r_square))


# =========================
# 3. 用一组新数据进行预测
# =========================
# 特征顺序必须和 Boston 数据集列顺序一致：
# CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT

x_test = np.array([
    [0.005, 16, 2, 0.0, 0.7, 6, 70, 4, 1.0, 297, 16, 398, 5]
])

pred_price = method.predict(x_test)

print('\n预测结果：')
print(pred_price)


# =========================
# 4. 使用 statsmodels 进行 OLS 回归分析
# =========================
# statsmodels 默认没有截距项，所以需要手动添加常数项

x_add1 = sm.add_constant(x)

model_1 = sm.OLS(y, x_add1).fit()

print('\n使用 statsmodels 建立完整模型：')
print(model_1.summary())


# =========================
# 5. 删除 P 值较大的属性 INDUS 和 AGE
# =========================

x_drop = x.copy()

x_drop.drop(
    ['INDUS', 'AGE'],
    axis=1,
    inplace=True
)

x_add2 = sm.add_constant(x_drop)

model_2 = sm.OLS(y, x_add2).fit()

print('\n删除 INDUS 和 AGE 后重新建模：')
print(model_2.summary())


# =========================
# 6. 查看模型参数
# =========================

print('\n删除两个属性后模型参数：')
print(model_2.params)