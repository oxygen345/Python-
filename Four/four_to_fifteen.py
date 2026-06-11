from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


# =========================
# 例4-15 利用互信息法对鸢尾花分类
# =========================

# 加载鸢尾花数据集
iris = load_iris()

# 特征数据
x = iris.data

# 标签数据
y = iris.target

# 返回每个特征与标签的互信息统计量
x_y_result = mutual_info_classif(x, y)

print('每个特征的互信息值：')
print(x_y_result)

# 筛选互信息量最大的2个特征
data_iris = SelectKBest(
    mutual_info_classif,
    k=2
).fit_transform(x, y)

print('筛选后的特征数据：')
print(data_iris)