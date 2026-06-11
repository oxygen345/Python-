# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# =========================
# 1. 构造训练数据
# =========================

data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年',
           '中年', '中年', '中年', '中年', '中年',
           '老年', '老年', '老年', '老年', '老年'],

    '收入': ['高', '高', '高', '中', '低',
           '高', '高', '中', '低', '低',
           '高', '中', '中', '低', '低'],

    '是否有房': ['否', '否', '是', '否', '否',
              '否', '否', '是', '是', '是',
              '是', '是', '否', '否', '否'],

    '信用情况': ['一般', '好', '好', '一般', '一般',
              '一般', '好', '好', '非常好', '非常好',
              '非常好', '好', '好', '非常好', '一般'],

    '是否同意贷款': ['否', '否', '是', '否', '否',
                '否', '否', '是', '是', '是',
                '是', '是', '是', '是', '否']
}

df = pd.DataFrame(data)

print("原始数据：")
print(df)


# =========================
# 2. 特征和标签分离
# =========================

X = df[['年龄', '收入', '是否有房', '信用情况']]
y = df['是否同意贷款']


# =========================
# 3. 将文字特征转换成数字
# =========================

encoders = {}

X_encoded = X.copy()

for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column])
    encoders[column] = le

# 标签也要转换成数字
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n转换后的特征数据：")
print(X_encoded)

print("\n转换后的标签：")
print(y_encoded)


# =========================
# 4. 构造 ID3 决策树模型
# =========================

# criterion='entropy' 表示使用信息熵，相当于使用信息增益思想
id3_tree = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0
)

# 训练模型
id3_tree.fit(X_encoded, y_encoded)


# =========================
# 5. 预测一个新的个人情况
# =========================

# 新用户情况：
# 年龄：青年
# 收入：中
# 是否有房：否
# 信用情况：好

new_person = pd.DataFrame({
    '年龄': ['青年'],
    '收入': ['中'],
    '是否有房': ['否'],
    '信用情况': ['好']
})

# 把新用户的文字信息转换成数字
new_person_encoded = new_person.copy()

for column in new_person.columns:
    new_person_encoded[column] = encoders[column].transform(new_person[column])

# 预测
prediction = id3_tree.predict(new_person_encoded)

# 把预测结果从数字转换回文字
result = label_encoder.inverse_transform(prediction)

print("\n新用户情况：")
print(new_person)

print("\n预测结果：")
print("是否同意贷款：", result[0])


# =========================
# 6. 可视化决策树
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei']      # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号

plt.figure(figsize=(14, 8))

plot_tree(
    id3_tree,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("ID3 决策树模型")
plt.show()