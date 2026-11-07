# 机器学习课程实验（SCUT）

华南理工大学《机器学习》课程四次实验的完整实现与报告。每个实验都提供 **scikit-learn 调包实现** 与 **纯 numpy 自实现** 两套代码，并把两者的训练过程、参数、预测结果放在一起对比，验证自实现与成熟库的等价性（或解释差异来源）。

- 四个 notebook：`exp1.ipynb` ~ `exp4.ipynb`（均已执行、保留输出）
- 八个自实现模型：`models/exp1` ~ `models/exp4`，接口与 scikit-learn 对齐，可直接互换
- 一份实验报告：`实验报告/MLSY吴隆权2024668615220103.docx`

## 目录结构

```
machine_learning/
├── exp1.ipynb                       # 实验一：逻辑斯蒂回归与线性判别分析
├── exp2.ipynb                       # 实验二：支持向量机与感知机
├── exp3.ipynb                       # 实验三：贝叶斯分类与 BP 算法
├── exp4.ipynb                       # 实验四：集成模型与决策树
├── models/                          # 纯 numpy 自实现模型
│   ├── exp1/sigmoid.py              #   Sigmoid + 逻辑斯蒂回归（牛顿法/梯度下降）
│   ├── exp1/LDA.py                  #   线性判别分析（Fisher 准则 + 广义特征值）
│   ├── exp2/svm.py                  #   支持向量机（SMO 求解对偶问题，支持多种核）
│   ├── exp2/perceptron.py           #   感知机（误分类驱动）
│   ├── exp3/naive_bayes.py          #   多项式朴素贝叶斯（支持稀疏输入）
│   ├── exp3/bp_network.py           #   单隐藏层 BP 网络（ReLU + Adam）
│   ├── exp4/adaboost.py             #   决策树桩 + AdaBoost
│   └── exp4/decision_tree.py        #   CART 决策树（熵/基尼、类别权重）
├── data/                            # 实验数据
│   ├── exp1/Simmons.xlsx
│   ├── exp2/auto-mpg.data, diabetes.csv
│   ├── exp3/winequality-red.csv, email/{ham,spam}/*.txt
│   └── exp4/Credit_card.csv, Credit_card_label.csv, Florence.xlsx
├── 实验报告/
│   └── MLSY吴隆权2024668615220103.docx
├── 上机内容/                        # 四次上机题目
├── homework/                        # 课后作业（chapter2 ~ chapter5）
├── 课程重点/                        # 期末复习提纲
├── PPT/                             # 课程课件（第 1~5 章）
├── 资料/                            # 参考书
├── requirements.txt                 # 依赖清单（pip / uv 均可）
├── pyproject.toml / uv.lock         # uv 工程配置
├── .gitignore                       # 忽略虚拟环境、缓存等
└── README.md
```

## 实验一览

| 实验 | 主题 | 数据集 | 调包模型 | 自实现模型 |
| --- | --- | --- | --- | --- |
| 1 | 逻辑斯蒂回归、线性判别分析 | Simmons（100 条，4 字段） | `LogisticRegression`、`LinearDiscriminantAnalysis` | `models/exp1/sigmoid.py`、`LDA.py` |
| 2 | 支持向量机、感知机 | auto-mpg（392 条）、diabetes（768 条） | `SVC`（RBF 核）、`Perceptron` | `models/exp2/svm.py`、`perceptron.py` |
| 3 | 朴素贝叶斯、BP 神经网络 | 邮件（50 封）、红酒品质（1599 条） | `MultinomialNB`、`MLPClassifier` | `models/exp3/naive_bayes.py`、`bp_network.py` |
| 4 | AdaBoost、决策树 | 信用卡（1548 条）、Florence（4000 条） | `AdaBoostClassifier`、`DecisionTreeClassifier` | `models/exp4/adaboost.py`、`decision_tree.py` |

### 主要结果（调包 vs 自实现）

| 实验 | 指标 | 调包实现 | 自实现 |
| --- | --- | --- | --- |
| 1 | 逻辑斯蒂回归准确率 | 60.00% | 60.00%（牛顿法 4 轮 / 梯度下降 1432 轮），系数最大差 1.4×10⁻⁵ |
| 1 | LDA 准确率 | 100.00% | 100.00%，预测逐样本一致，投影相关系数 1.0 |
| 2 | SVM 准确率 | 89.87%（89 个支持向量） | 89.87%（89 个支持向量），预测 100% 一致 |
| 2 | 感知机准确率 | 72.73% | 68.83%，预测一致率 79.22%（感知机解不唯一） |
| 3 | 朴素贝叶斯 5 折准确率 | 94.00% | 94.00%，每折完全相同，对数条件概率差 0 |
| 3 | BP（2 个神经元）误差率 | 54.17%（退化为常数分类器） | 25.62% |
| 4 | AdaBoost 准确率 | 88.17% | 87.96%，预测一致率 99.78% |
| 4 | 决策树准确率 | 55.63% | 56.25%，预测一致率 99.38% |

## 自实现模型

所有自实现模型只依赖 `numpy`（朴素贝叶斯与 AdaBoost 另用 `scipy.sparse` 兼容稀疏输入），接口与 scikit-learn 同名，因此 notebook 中可以直接替换实例做对比：

| 文件 | 类 | 主要接口 | 求解思路 |
| --- | --- | --- | --- |
| `models/exp1/sigmoid.py` | `LogisticRegression` | `fit / predict / predict_proba / decision_function`、`coef_ / intercept_ / loss_history_` | 带 L2 正则的极大似然；牛顿法（解析梯度 + 海森矩阵）或梯度下降 |
| `models/exp1/LDA.py` | `LinearDiscriminantAnalysis` | `fit / transform / predict / predict_proba`、`means_ / priors_ / scalings_` | 类内散度 Cholesky 白化后解广义特征值问题，再按高斯判别函数分类 |
| `models/exp2/svm.py` | `SVC` | `fit / predict / decision_function`、`support_ / support_vectors_ / dual_coef_` | SMO 算法求解对偶问题（最大违反 KKT 的工作集选择），支持 linear / poly / rbf 核 |
| `models/exp2/perceptron.py` | `Perceptron` | `fit / predict / decision_function`、`coef_ / intercept_` | 误分类驱动的在线学习，含早停 |
| `models/exp3/naive_bayes.py` | `MultinomialNB` | `fit / predict / predict_proba / predict_log_proba`、`feature_log_prob_ / class_log_prior_` | 类先验 + 拉普拉斯平滑的类条件概率，对数域累加 |
| `models/exp3/bp_network.py` | `MLPClassifier` | `fit / predict / predict_proba`、`loss_curve_ / n_iter_` | 单隐藏层、ReLU、交叉熵、反向传播 + Adam |
| `models/exp4/adaboost.py` | `DecisionStump`、`AdaBoostClassifier` | `fit / predict / decision_function`、`estimators_ / estimator_weights_` | 加权错误率选桩，按 α 加权投票，指数损失更新样本权重 |
| `models/exp4/decision_tree.py` | `DecisionTreeClassifier` | `fit / predict / predict_proba`、`feature_importances_` | CART 贪心划分（熵/基尼），支持深度、叶节点样本数与类别权重 |

## 环境配置

- **Python 3.12**（开发环境为 3.12.3）
- 依赖见 [`requirements.txt`](requirements.txt)（版本与开发环境逐项一致），主要包括 numpy、pandas、scipy、scikit-learn、matplotlib、seaborn、openpyxl、jupyter，以及报告生成用的 python-docx、nbclient、nbformat、pillow

> `pyproject.toml` 只声明了运行 notebook 的主要依赖；生成报告还需要 python-docx。
> 因此推荐直接按 `requirements.txt` 安装，它包含全部依赖。

### 方式一：使用 uv（推荐）

```bash
uv venv                                    # 创建 .venv
uv pip install -r requirements.txt         # 安装全部依赖
```

### 方式二：使用 venv + pip

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始

```bash
# 启动 Jupyter（在项目根目录执行）
jupyter notebook
```

然后在浏览器中打开 `exp1.ipynb` ~ `exp4.ipynb`，逐个运行即可；也可用命令行一次性执行：

```bash
jupyter nbconvert --to notebook --execute exp1.ipynb --inplace
```

> 注意：**必须在项目根目录启动**。notebook 中使用的是相对路径 `data/...`，并通过 `sys.path.insert(0, ".")` 导入 `models` 包（`models/` 没有 `__init__.py`，依靠命名空间包机制）。

## 复现说明

- 所有 notebook 已保存执行输出，直接查看即可；重新运行可完整复现（涉及随机性的步骤统一使用 `random_state=42`）。
- 图形中的坐标轴、标题使用英文，是因为 notebook 默认字体不含中文字形、中文会显示成方块；控制台打印的结论仍为中文。
- 实验三的邮件数据用 `glob` 读取，文件顺序会影响训练/测试划分；样本只有 50 封，因此不同机器上每折准确率可能相差 10 个百分点（每折仅 10 封邮件），属于正常现象。

## 实验报告

`实验报告/MLSY吴隆权2024668615220103.docx` 由 python-docx + matplotlib 自动排版生成，正文约 90 页。每个实验按“实验目标 / 实验数据 / 实验要求 / 实验步骤”组织，实验步骤中每个要求都包含：

1. 【代码实现】先贴调包实现，再贴自实现（含模型源码）；
2. 【原理与推导】公式推导（matplotlib 渲染的编号公式）与原理示意图；
3. 【运行结果】两种实现的对比表、对比图与结论分析。

报告的生成脚本（`实验报告/build/`）与插图（`实验报告/素材/`）在整理仓库时被移除，但仍保留在 git 历史中，需要时用下面的命令取回：

```bash
git checkout 5db9055 -- 实验报告/build 实验报告/素材
```

## 其它目录

| 目录 | 说明 |
| --- | --- |
| `上机内容/` | 四次上机实验的题目文档 |
| `homework/` | 课后作业（第 2~5 章） |
| `课程重点/` | 期末考试复习提纲 |
| `PPT/` | 课程课件（第 1~5 章） |
| `资料/` | 机器学习公式详解等参考书 |

## 说明

本项目为课程作业，代码与报告仅供学习参考。
