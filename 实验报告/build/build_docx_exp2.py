# -*- coding: utf-8 -*-
"""实验二实验报告生成脚本（支持向量机与感知机）。

风格与实验一定稿一致：
  * 只撰写"4.实验步骤"部分，其余章节照抄实验二报告题目文件
  * 每项要求按【实验要求】框 → 【代码实现】代码框 → 【原理与推导】 → 【运行结果】组织
  * 不写 a/b/c/d 小标题；图注不带来源/主题后缀
  * 代码直接从 exp2.ipynb 逐字符读取，保证与 notebook 一致
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docx_kit as K

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
ASSETS = os.path.join(REPORT_DIR, "素材", "exp2")
NB = json.load(open(os.path.join(PROJ_DIR, "exp2.ipynb"), encoding="utf-8"))
OUT_PATH = os.path.join(REPORT_DIR, "实验二 支持向量机与感知机.docx")

NUM = K.Numbering("2")


def code(i):
    return "".join(NB["cells"][i]["source"])


def out_text(i):
    parts = []
    for o in NB["cells"][i].get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o["text"]))
        elif o.get("output_type") == "execute_result":
            parts.append("".join(o["data"].get("text/plain", [])))
    return "".join(parts).strip("\n")


def J(*parts):
    return os.path.join(ASSETS, *parts)


def eq(doc, key):
    return K.add_eq(doc, J("formula", key + ".png"), NUM)


def fig(doc, rel, caption):
    return K.add_fig(doc, J(*rel.split("/")), caption, NUM)


def build():
    doc = K.new_doc()
    K.cover_table(doc, date="2025.11.21")
    K.doc_title(doc, "实验二　支持向量机及核函数的使用")

    # ================================================================= 一
    K.h1(doc, "一、支持向量机")
    K.h2(doc, "1.实验目标")
    K.body(doc, "a.理解支持向量机的原理与应用方法")
    K.body(doc, "b.学习如何使用核函数将分类任务转化为线性或非线性可分的问题。")

    K.h2(doc, "2.实验数据")
    K.body(doc, "将 Auto 数据集中的 mpg 按照中位数分为两类，新增一个变量 grade，"
                "并用 0 和 1 分别表示。")

    K.h2(doc, "3.实验要求")
    for s in ["a.编写Python代码实现支持向量机。",
              "b.加载数据集后，进行基本的数据清洗和处理.",
              "c.将数据集分为训练集和测试集，使用支持向量机模型进行训练，"
              "选择合适的核函数如RBF核函数。（可以尝试使用其他核函数）",
              "d.在测试集上进行预测，并计算模型的准确率、混淆矩阵、精确率等指标。",
              "e.绘制分类结果的混淆矩阵图。"]:
        K.body(doc, s)

    K.h2(doc, "4.实验步骤")
    K.body(doc, "本实验在 Python 3.12 环境下完成：pandas 负责数据读取与整理，matplotlib 负责"
                "绘图，scikit-learn 提供支持向量机、感知机与各类评估指标的实现。首先导入"
                "全部依赖库：")
    K.code_box(doc, code(1))

    # ---- a
    K.req_box(doc, "a.编写Python代码实现支持向量机。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(3))
    K.label(doc, "【原理与推导】")
    K.body(doc, "支持向量机的基本思想是在特征空间中寻找一个既能分开两类样本、又使间隔尽可能"
                "大的超平面，该超平面可写为：")
    e1 = eq(doc, "f2_hyperplane")
    K.body(doc, f"如式 {e1} 所示，样本到超平面的距离由 |wᵗx + b| / ‖w‖ 决定。若要求所有样本都"
                "被正确分类且位于间隔之外，即满足 yᵢ(wᵗxᵢ + b) ≥ 1，则两类之间的间隔宽度为：")
    e2 = eq(doc, "f2_margin")
    K.body(doc, f"式 {e2} 表明间隔只与权向量的模长有关，模长越小间隔越大。于是学习问题转化为"
                "一个带线性约束的凸二次规划：")
    e3 = eq(doc, "f2_hard")
    K.body(doc, f"式 {e3} 即硬间隔支持向量机。决定间隔的只有落在间隔边界上的少数样本，它们"
                "称为支持向量；删除其他样本不影响解，这正是支持向量机解的稀疏性。")
    fig(doc, "theory/fig2_svm_margin.jpg", "最大间隔超平面与支持向量")
    K.body(doc, "实际数据往往线性不可分，为每个样本引入松弛变量 ξᵢ 与惩罚系数 C，允许少量样本"
                "进入间隔甚至被错分，得到软间隔支持向量机：")
    e4 = eq(doc, "f2_soft")
    K.body(doc, f"式 {e4} 中 C 控制对违反间隔的惩罚力度：C 越大，模型越不容忍错分，间隔被压缩"
                "甚至过拟合；C 越小，模型允许更多样本进入间隔，间隔更宽、泛化性更好。软间隔"
                "目标也可以写成合页损失加 L2 正则的形式：")
    e5 = eq(doc, "f2_hinge")
    fig(doc, "theory/fig2_soft_margin_C.jpg", "惩罚系数 C 对间隔宽度与容错能力的影响")
    K.body(doc, "代码中 SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\") 即建立一个以高斯核为核函数、"
                "惩罚系数取 1 的支持向量机。")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建模型对象，尚未训练，因此没有输出。")

    # ---- b
    K.req_box(doc, "b.加载数据集后，进行基本的数据清洗和处理.")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(5))
    K.body(doc, "原始数据以空白字符分隔，且缺失值以 “?” 表示，因此读取时显式指定分隔符与缺失值"
                "标记；数据清洗与标签构造如下：")
    K.code_box(doc, code(6))
    K.code_box(doc, code(7))
    K.label(doc, "【原理与推导】")
    K.body(doc, "Auto MPG 数据集共 398 条样本、9 个字段（含汽车名称）。其中 horsepower 有 6 个"
                "缺失值，若直接建模会因缺失而报错或引入偏差，故先删除含有缺失的样本，处理"
                "后剩余 392 条。")
    K.body(doc, "本实验是回归目标转分类任务的典型做法：以 mpg 的中位数为阈值，大于等于中位数"
                "的样本记为 1（高油耗里程），其余记为 0（低油耗里程）。由于阈值取中位数，"
                "变换后两类样本数量完全相同（各 196 条），是一组天然平衡的二分类数据，"
                "评估时不必担心类别偏斜。")
    fig(doc, "theory/fig2_mpg_binning.jpg", "按 mpg 中位数构造二分类标签")
    K.label(doc, "【运行结果】")
    K.body(doc, "读取后打印数据结构：horsepower 列的非空数量为 392，缺失的 6 条即被识别为 NaN；"
                "构造标签后两类的样本数均为 196。")
    K.out_box(doc, out_text(5))
    K.out_box(doc, out_text(7))

    # ---- c
    K.req_box(doc, "c.将数据集分为训练集和测试集，使用支持向量机模型进行训练，"
                   "选择合适的核函数如RBF核函数。（可以尝试使用其他核函数）")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(9))
    K.body(doc, "支持向量机的间隔与核函数都建立在欧氏距离之上，特征量纲不同会使距离被量纲大的"
                "特征主导，因此训练前先做标准化：")
    K.code_box(doc, code(10))
    K.body(doc, "在标准化后的训练集上拟合模型：")
    K.code_box(doc, code(11))
    K.label(doc, "【原理与推导】")
    K.body(doc, "特征矩阵取 cylinders、displacement、horsepower、weight、acceleration、"
                "model_year、origin 共 7 个数值特征（汽车名称为文本，不参与建模），并采用"
                "分层采样按 8 : 2 划分训练集与测试集，使两类的比例在训练集与测试集中保持"
                "一致。标准化按下式进行，且均值与标准差只在训练集上估计，再用于测试集：")
    e6 = eq(doc, "f2_scale")
    K.body(doc, f"式 {e6} 使每个特征都服从均值 0、方差 1 的分布，各特征对距离的贡献因此可比。")
    K.body(doc, "当样本在原始空间中线性不可分时，可以通过核函数把样本隐式地映射到高维特征"
                "空间，使其在新空间中线性可分，而计算仍然只涉及原始空间的内积：")
    e7 = eq(doc, "f2_kernel_def")
    e8 = eq(doc, "f2_rbf")
    e9 = eq(doc, "f2_poly")
    K.body(doc, f"式 {e7} 是核函数的定义；式 {e8} 为高斯（RBF）核，γ 越大单个样本的影响范围"
                f"越小；式 {e9} 为多项式核。下图以同心圆数据说明核技巧的作用。")
    fig(doc, "theory/fig2_kernel_trick.jpg", "核技巧：二维线性不可分的数据在三维空间中线性可分")
    K.body(doc, "为验证核函数的选择，在同一份数据上分别使用线性核、三次多项式核与高斯核，"
                "并比较测试集准确率（为便于在二维平面上绘制决策边界，这里只取 weight 与 "
                "horsepower 两个特征，因此准确率与使用全部 7 个特征时不同）：")
    fig(doc, "theory/fig2_kernel_compare.jpg", "不同核函数的决策边界与测试集准确率对比")
    K.body(doc, "高斯核中 γ 的取值直接影响模型的复杂度：γ 过小会使决策边界过于平滑而欠拟合，"
                "γ 过大则会使每个样本各自形成一个小区域而过拟合：")
    fig(doc, "theory/fig2_gamma_effect.jpg", "核参数 γ 对决策边界与拟合程度的影响")
    K.body(doc, "为进一步确定合适的参数组合，对惩罚系数 C 与核参数 γ 做 5 折交叉验证网格搜索，"
                "结果如下（图中红框为最优组合）：")
    fig(doc, "theory/fig2_param_grid.jpg", "C 与 γ 的网格搜索结果（5 折交叉验证准确率）")
    K.body(doc, "上述过程对应支持向量机的对偶形式。对软间隔问题引入拉格朗日乘子 αᵢ 后，原问题"
                "可转化为只含内积的对偶问题：")
    e10 = eq(doc, "f2_lagrange")
    e11 = eq(doc, "f2_dual")
    K.body(doc, f"式 {e10} 为拉格朗日函数，式 {e11} 为对偶问题。其 KKT 条件给出两个重要结论："
                "只有 αᵢ > 0 的样本（支持向量）参与权向量的构造，且权向量是支持向量的线性"
                "组合：")
    e12 = eq(doc, "f2_kkt")
    K.body(doc, "预测时只需计算待判样本与支持向量的核函数值即可：")
    e13 = eq(doc, "f2_decision")
    K.body(doc, f"式 {e13} 说明：支持向量机的计算量与支持向量个数成正比，而核函数使式中的"
                "内积可以在高维空间中隐式计算，这正是核方法的价值所在。")
    K.label(doc, "【运行结果】")
    K.body(doc, "模型拟合完成，输出的是所用估计器的参数：")
    K.out_box(doc, out_text(11))

    # ---- d
    K.req_box(doc, "d.在测试集上进行预测，并计算模型的准确率、混淆矩阵、精确率等指标。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(13))
    K.code_box(doc, code(14))
    K.code_box(doc, code(15))
    K.code_box(doc, code(16))
    K.label(doc, "【原理与推导】")
    K.body(doc, "分类任务的准确率定义为判对样本占样本总数的比例：")
    e14 = eq(doc, "f2_acc")
    K.body(doc, "准确率在类别不平衡时容易高估模型效果，因此还需要混淆矩阵以及由它导出的"
                "查准率、查全率与 F1 分数：")
    e15 = eq(doc, "f2_prf")
    K.body(doc, f"式 {e15} 中，查准率 P 衡量“被判为正例的样本中真正例的比例”，查全率 R 衡量"
                "“真正的正例中被找出来的比例”，F1 是二者的调和平均。多分类或需要综合两类"
                "表现时，还可使用宏平均与加权平均：")
    e16 = eq(doc, "f2_macro")
    K.body(doc, f"式 {e16} 中宏平均对每一类同等看待，加权平均则按各类样本数加权。二分类的"
                "混淆矩阵中，TP 表示高油耗样本被判为高油耗的数量，FN 为漏判数量，FP 为误判"
                "数量，TN 为低油耗样本被判为低油耗的数量。")
    K.label(doc, "【运行结果】")
    K.body(doc, "在 79 个测试样本上，模型准确率为 89.87%：")
    K.out_box(doc, out_text(14))
    K.body(doc, "混淆矩阵显示：40 个低 mpg 样本中有 34 个判对、6 个判错，39 个高 mpg 样本中"
                "有 37 个判对、2 个判错：")
    K.out_box(doc, out_text(15))
    K.body(doc, "分类报告给出了每一类的查准率、查全率与 F1 分数，宏平均与加权平均均为 0.90，"
                "两类的表现比较均衡：")
    K.out_box(doc, out_text(16))

    # ---- e
    K.req_box(doc, "e.绘制分类结果的混淆矩阵图。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(18))
    K.label(doc, "【原理与推导】")
    K.body(doc, "混淆矩阵的可视化把四个计数放到同一张热力图上：对角线颜色越深说明判对的样本"
                "越多，非对角线上的格子则对应两类错误。相比只看准确率，它能直观暴露模型是"
                "偏向漏判还是偏向误判。")
    K.label(doc, "【运行结果】")
    K.add_fig(doc, J("notebook_out", "cell18.jpg"), "SVM 分类结果的混淆矩阵", NUM)

    # ================================================================= 二
    doc.add_page_break()
    K.h1(doc, "二、感知机")
    K.h2(doc, "1.实验目标")
    K.body(doc, "a.理解感知机的基本原理及其应用。")
    K.body(doc, "b.掌握感知机模型的构建和评估方法。")

    K.h2(doc, "2.实验数据")
    K.body(doc, "使用糖尿病数据集(diabetes.csv)，该数据集包含多个特征，"
                "用于预测患者是否患有糖尿病。")

    K.h2(doc, "3.实验要求")
    for s in ["a.编写Python代码实现感知机模型。",
              "b.实现数据的预处理，使用感知机模型进行糖尿病数据的分类。",
              "c.输出模型的准确率、混淆矩阵。",
              "d.可视化特征权重，以分析模型性能和特征重要性。"]:
        K.body(doc, s)

    K.h2(doc, "4.实验步骤")

    # ---- a
    K.req_box(doc, "a.编写Python代码实现感知机模型。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(22))
    K.label(doc, "【原理与推导】")
    K.body(doc, "感知机是最早的线性分类模型，它直接对样本输出类别而无需概率解释：")
    e17 = eq(doc, "f2_perc_model")
    K.body(doc, f"式 {e17} 中 sign(·) 为符号函数。感知机采用误分类驱动的学习策略：只统计被"
                "当前模型判错的样本集合 M，以误分类样本到超平面的距离之和（略去 1/‖w‖ 的"
                "尺度因子）作为损失函数：")
    e18 = eq(doc, "f2_perc_loss")
    K.body(doc, f"式 {e18} 对 w 与 b 求梯度：")
    e19 = eq(doc, "f2_perc_grad")
    K.body(doc, f"式 {e19} 给出随机梯度下降的更新规则，即每发现一个误分类样本就沿其方向"
                "调整一次参数：")
    e20 = eq(doc, "f2_perc_update")
    K.body(doc, "代码中 Perceptron(max_iter=1000, tol=1e-3, random_state=42) 表示最多迭代 "
                "1000 轮，若损失的下降幅度连续小于 1e-3 则提前停止。下图给出决策边界随迭代"
                "的变化以及训练错分率曲线：")
    fig(doc, "theory/fig2_perceptron_update.jpg",
        "感知机决策边界的迭代过程与训练错分率变化")
    K.body(doc, "Novikoff 定理指出：若训练集线性可分，感知机一定能在有限步内收敛，且迭代次数"
                "存在上界：")
    e21 = eq(doc, "f2_novikoff")
    K.body(doc, f"式 {e21} 中 γ 为样本到最优超平面的最小函数间隔，R 为样本的最大模长。需要"
                "注意的是，感知机只要求把样本分开，满足条件的解并不唯一，它也不会像支持向量"
                "机那样去最大化间隔。下图对比了两种模型在同一数据上的决策边界。")
    fig(doc, "theory/fig2_perceptron_vs_svm.jpg",
        "感知机与线性 SVM 的决策边界对比（仅取两个特征以便可视化）")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建感知机对象，尚未训练，因此没有输出。")

    # ---- b
    K.req_box(doc, "b.实现数据的预处理，使用感知机模型进行糖尿病数据的分类。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(20))
    K.body(doc, "读取数据后，先观察数据结构，再划分训练集与测试集：")
    K.code_box(doc, code(24))
    K.body(doc, "感知机同样对特征尺度敏感，故先标准化：")
    K.code_box(doc, code(25))
    K.code_box(doc, code(26))
    K.body(doc, "在测试集上进行预测：")
    K.code_box(doc, code(27))
    K.label(doc, "【原理与推导】")
    K.body(doc, "糖尿病数据集共 768 条样本、8 个数值特征（怀孕次数、血糖、血压、皮褶厚度、"
                "胰岛素、BMI、糖尿病家族史函数、年龄）与 1 个标签 Outcome，其中未患病 500 例、"
                "患病 268 例。若模型不加学习、一律预测为未患病，准确率也能达到：")
    e22 = eq(doc, "f2_baseline")
    K.body(doc, f"式 {e22} 即多数类基线，该数据上为 65.1%。它是评价模型是否真正学到信息的"
                "重要参照。")
    fig(doc, "theory/fig2_class_balance.jpg", "糖尿病数据集的类别分布与多数类基线")
    K.body(doc, "需要说明的是，数据表中没有显式的缺失值，但 Glucose、BloodPressure、"
                "SkinThickness、Insulin、BMI 等字段存在取值为 0 的记录（如 SkinThickness 有 "
                "227 条、Insulin 有 374 条），这些取值在生理上不可能出现，应视为缺失。本实验"
                "按 notebook 的处理方式保留原始取值，未做插补，这也是模型性能受限的原因之一。")
    K.body(doc, f"训练前对特征做标准化（同式 {e6}），使各特征对权重的贡献可比；随后按 8 : 2 "
                "分层划分训练集与测试集，以保证测试集中患病样本的比例与整体一致。")
    K.label(doc, "【运行结果】")
    K.body(doc, "读取后打印数据结构：共 768 条样本、9 个字段，无缺失值；模型拟合完成后的"
                "参数为：")
    K.out_box(doc, out_text(20))
    K.out_box(doc, out_text(26))

    # ---- c
    K.req_box(doc, "c.输出模型的准确率、混淆矩阵。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(29))
    K.code_box(doc, code(30))
    K.code_box(doc, code(31))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"评价指标与支持向量机部分相同，仍使用准确率（式 {e14}）、混淆矩阵以及查准率、"
                f"查全率与 F1 分数（式 {e15}）。由于该数据存在类别不平衡，本次评估需要把结果"
                f"与多数类基线（式 {e22}，65.1%）对照来看。")
    fig(doc, "concept/fig2_perceptron_cm.jpg", "感知机在糖尿病测试集上的混淆矩阵")
    K.label(doc, "【运行结果】")
    K.body(doc, "模型准确率为 72.73%，仅比多数类基线高 7.6 个百分点：")
    K.out_box(doc, out_text(29))
    K.body(doc, "混淆矩阵显示：100 个未患病样本中有 80 个判对、20 个误判为患病；54 个患病"
                "样本中只有 32 个被识别出来，漏判 22 个：")
    K.out_box(doc, out_text(30))
    K.body(doc, "分类报告进一步说明问题所在：患病类的查全率只有 0.59，即约四成患者被漏判，"
                "其 F1 分数仅 0.60，明显低于未患病类的 0.79。若只看准确率会高估模型效果，"
                "这一现象正体现了类别不平衡场景下使用多指标评估的必要性：")
    K.out_box(doc, out_text(31))

    # ---- d
    K.req_box(doc, "d.可视化特征权重，以分析模型性能和特征重要性。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(33))
    K.code_box(doc, code(34))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"由式 {e17} 可知，特征权重 w 决定了每个特征对判决的方向与强度：权重为正表示"
                "该特征取值越大越倾向于判为患病，为负则相反；绝对值越大说明该特征对分类的"
                "影响越强。由于训练前所有特征都已标准化，不同特征之间的权重具有可比性，"
                "因此可以直接用权重的绝对值来衡量特征重要性。")
    K.body(doc, "从结果看，Glucose（血糖）的权重最大，与临床上血糖是糖尿病主要判别指标的"
                "事实一致；BMI 与 Age 也给出正的贡献；而 SkinThickness、Insulin 等特征的"
                "权重接近 0，说明在线性模型下它们对判决的贡献有限——这也与 b 中提到的这些"
                "字段存在大量 0 值（相当于缺失）有关。")
    K.label(doc, "【运行结果】")
    K.add_fig(doc, J("notebook_out", "cell34.jpg"), "感知机各特征的权重（按绝对值升序排列）", NUM)

    doc.save(OUT_PATH)
    print("已生成：", OUT_PATH)
    print(f"插图 {NUM.fig} 张，公式 {NUM.eq} 个")


if __name__ == "__main__":
    build()
