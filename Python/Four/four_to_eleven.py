import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 创建模拟数据
# =========================

np.random.seed(123)

onp_df = pd.DataFrame({
    'n_tokens_content': np.random.exponential(scale=500, size=30000)
})


# =========================
# 2. 标准化
# =========================

onp_df['standardized_n'] = preproc.StandardScaler().fit_transform(
    onp_df[['n_tokens_content']]
)


# =========================
# 3. Min-Max 缩放
# =========================

onp_df['minmax_n'] = preproc.minmax_scale(
    onp_df['n_tokens_content']
)


# =========================
# 4. L2 归一化
# =========================

onp_df['l2_normalized_n'] = preproc.normalize(
    onp_df[['n_tokens_content']],
    axis=0
)


# =========================
# 5. 绘图
# =========================

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 12))
plt.subplots_adjust(wspace=0, hspace=0.5)

ax1.hist(onp_df['n_tokens_content'], bins=100, color='skyblue', edgecolor='black')
ax1.set_ylabel('分享次数')
ax1.set_title('原始单词量数据')

ax2.hist(onp_df['standardized_n'], bins=100, color='orange', edgecolor='black')
ax2.set_ylabel('分享次数')
ax2.set_title('标准化数据')

ax3.hist(onp_df['minmax_n'], bins=100, color='lightgreen', edgecolor='black')
ax3.set_ylabel('分享次数')
ax3.set_title('MinMax缩放数据')

ax4.hist(onp_df['l2_normalized_n'], bins=100, color='salmon', edgecolor='black')
ax4.set_ylabel('分享次数')
ax4.set_title('L2归一化数据')

plt.show()