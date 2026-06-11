# Python 机器学习算法及其应用

Python 机器学习课程代码仓库，涵盖经典机器学习算法的原理实现与实战应用。

---

## 📚 目录结构

| 章节 | 目录 | 说明 | 文件数 |
|------|------|------|--------|
| 第三章 | [three/](three/) | 感知机、决策树（ID3） | 8 |
| 第四章 | [Four/](Four/) | 数据预处理、缺失值填补、拉格朗日插值 | 16 |
| 第五章 | [Five/](Five/) | 主成分分析（PCA）降维 | 8 |
| 第六章 | [six/](six/) | 集成学习、投票分类器、ROC 曲线 | 6 |

---

## 🧪 算法概览

### 第三章 · 分类基础

- **感知机（Perceptron）** — 基于鸢尾花数据集的二分类/多分类，含决策区域可视化
- **ID3 决策树** — 基于信息熵（entropy）的决策树，含中文贷款审批示例与树结构绘图

### 第四章 · 数据预处理

- **缺失值填补** — 均值、中位数、众数、临近值（ffill）四种策略对比
- **拉格朗日插值法** — 自定义插值函数填补缺失值，含 KDE 分布对比图

### 第五章 · 降维

- **PCA 主成分分析** — 从零实现（基于梯度上升法求第一主成分），支持 transform / inverse_transform

### 第六章 · 集成学习

- **多投票机制组合分类器（VotingClassifier）** — 逻辑回归 + 决策树 + KNN 软投票
- **ROC 曲线与 AUC** — 10 折交叉验证，对比单个模型与集成模型的分类效果
- **决策区域可视化** — 四种分类器在同一坐标系下的决策边界对比

---

## 🛠 环境要求

- Python 3.7+
- NumPy
- pandas
- scikit-learn
- matplotlib
- SciPy

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

> 中文字体：代码中使用 `SimHei` / `Microsoft YaHei`，Windows 系统可直接显示中文；Linux / macOS 请自行安装对应中文字体或修改 `plt.rcParams['font.sans-serif']`。

---

## 🚀 使用方法

直接运行任意 `.py` 文件即可：

```bash
# 运行第三章感知机示例
python three/P64.py

# 运行第六章集成学习示例
python six/six_to_one.py
```

每个脚本都是**自包含**的，无需额外数据集（内部使用 sklearn 内置的 iris 数据集或手工构造数据）。

---

## 📄 License

课程学习用途。
