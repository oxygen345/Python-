import pandas as pd
import numpy as np
from scipy import stats


# =========================
# 例4-14 以掷骰子为例
# 卡方拟合优度检验
# =========================

# 观测频数：实际掷骰子120次后，每个点数出现的次数
observed_pd = pd.DataFrame(
    ['1点'] * 23 +
    ['2点'] * 20 +
    ['3点'] * 18 +
    ['4点'] * 19 +
    ['5点'] * 24 +
    ['6点'] * 16
)

# 期望频数：如果骰子是均匀的，每个点数期望出现20次
expected_pd = pd.DataFrame(
    ['1点'] * 20 +
    ['2点'] * 20 +
    ['3点'] * 20 +
    ['4点'] * 20 +
    ['5点'] * 20 +
    ['6点'] * 20
)

# 生成频数表
observed_table = pd.crosstab(
    index=observed_pd[0],
    columns='count'
)

expected_table = pd.crosstab(
    index=expected_pd[0],
    columns='count'
)

print('观测频数表：')
print(observed_table)

print('----------------')

print('期望频数表：')
print(expected_table)


# =========================
# 方法一：手动计算卡方统计量
# =========================

observed = observed_table
expected = expected_table

chi_squared_stat = (((observed - expected) ** 2) / expected).sum()

print('开始计算卡方值')
print(chi_squared_stat)


# 显著性水平 alpha = 0.05
# 自由度 df = 类别数 - 1 = 6 - 1 = 5
crit = stats.chi2.ppf(q=0.95, df=5)

print('临界值：')
print(crit)

# 计算 P 值
P_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df=5)

print('P_value：')
print(P_value)


# =========================
# 方法二：直接使用 scipy 的 chisquare 函数
# =========================

result = stats.chisquare(
    f_obs=observed,
    f_exp=expected
)

print('scipy计算结果：')
print(result)