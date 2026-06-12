import zipfile
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# 设置中文字体
# =========================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 自动定位数据文件
# =========================

base_dir = Path(__file__).resolve().parent

csv_path = base_dir / "jena_climate_2009_2016.csv"
zip_path = base_dir / "jena_climate_2009_2016.csv.zip"


# 如果只有 zip 文件，没有 csv 文件，就自动解压
if not csv_path.exists():
    if zip_path.exists():
        print("发现压缩包，正在解压...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(base_dir)
        print("解压完成")
    else:
        raise FileNotFoundError(
            "没有找到 jena_climate_2009_2016.csv 或 jena_climate_2009_2016.csv.zip"
        )


# =========================
# 2. 读取数据
# =========================

data_train_set = pd.read_csv(csv_path)

print("数据前5行：")
print(data_train_set.head())


# =========================
# 3. 计算相关系数矩阵
# =========================
# Date Time 是时间字符串，不参与相关系数计算
# numeric_only=True 表示只计算数值列

d = data_train_set.corr(numeric_only=True)

print("\n相关系数矩阵：")
print(d)


# =========================
# 4. 绘制热力图
# =========================

plt.figure(figsize=(12, 12))

sns.heatmap(
    d,
    annot=True,
    vmax=1,
    vmin=-1,
    square=True,
    cmap="Reds",
    fmt=".2f",
    linewidths=0.5
)

plt.title("图7-6 热力图")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()