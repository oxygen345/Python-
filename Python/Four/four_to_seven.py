import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 创建示例数据
# =========================

excel_data = pd.DataFrame({
    'value': [58, -81, 75, 96, 99, -67, 100, 72, 75, 68, 89]
})

print('原始数据：')
print(excel_data)


# =========================
# 2. 使用 3σ 原则检测异常值
# =========================

def three_sigma(ser):
    """
    ser参数：被检测的数据，接收DataFrame的一列数据
    返回：异常值及其对应的行索引
    """

    # 计算平均值
    mean_data = ser.mean()

    # 计算标准差
    std_data = ser.std()

    # 小于 μ - 3σ 或大于 μ + 3σ 的数据均为异常值
    rule = (mean_data - 3 * std_data > ser) | (mean_data + 3 * std_data < ser)

    # 获取异常值的索引
    index = np.arange(ser.shape[0])[rule]

    # 获取异常值
    outliers = ser.iloc[index]

    return outliers


three_sigma_result = three_sigma(excel_data['value'])

print('\n3σ原则检测出的异常值：')
print(three_sigma_result)


# =========================
# 3. 绘制箱形图
# =========================

excel_data.boxplot(column='value')
plt.title('箱形图')
plt.show()


# =========================
# 4. 使用箱形图检测异常值
# =========================

def box_outliers(ser):
    """
    使用箱形图规则检测异常值
    """

    # 对待检测的数据集进行排序
    new_ser = ser.sort_values()

    # 判断数据总数量是奇数还是偶数
    if new_ser.count() % 2 == 0:
        # 计算 Q3、Q1、IQR
        Q3 = new_ser[int(len(new_ser) / 2):].median()
        Q1 = new_ser[:int(len(new_ser) / 2)].median()

    elif new_ser.count() % 2 != 0:
        Q3 = new_ser[int(len(new_ser) / 2 + 1):].median()
        Q1 = new_ser[:int(len(new_ser) / 2 - 1)].median()

    IQR = round(Q3 - Q1, 1)

    # 异常值判断规则
    rule = (round(Q3 + 1.5 * IQR, 1) < ser) | (round(Q1 - 1.5 * IQR, 1) > ser)

    # 获取异常值的索引
    index = np.arange(ser.shape[0])[rule]

    # 获取异常值
    outliers = ser.iloc[index]

    return outliers


box_result = box_outliers(excel_data['value'])

print('\n箱形图检测出的异常值：')
print(box_result)


# =========================
# 5. 删除异常值
# =========================

# 根据上面检测出的异常值行索引删除异常值
clean_data = excel_data.drop(box_result.index)

print('\n删除异常值后的数据：')
print(clean_data)

# 再次检测数据中是否还有异常值
print('\n删除后再次检测异常值：')
print(three_sigma(clean_data['value']))


# =========================
# 6. 替换异常值
# =========================

# 假设负值处理为0分，超过100分统一按100分计算
replace_data = excel_data.replace({
    -100: 0,
    200: 100
})

# 这里根据当前数据，直接把小于0的成绩替换为0
replace_data.loc[replace_data['value'] < 0, 'value'] = 0

# 大于100的成绩替换为100
replace_data.loc[replace_data['value'] > 100, 'value'] = 100

print('\n替换异常值后的数据：')
print(replace_data)

print('\n原索引1的数据：')
print(replace_data.loc[1])

print('\n原索引5的数据：')
print(replace_data.loc[5])