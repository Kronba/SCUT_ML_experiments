# -*- coding: utf-8 -*-
"""实验三：调包实现 + 自实现 的新版报告，写入 总实验报告 (Copy).docx。

替换原实验三章节；实验一、二、四保持不动。
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docx_kit as K
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
COPY = os.path.join(REPORT_DIR, "总实验报告 (Copy).docx")
ASSETS = os.path.join(REPORT_DIR, "素材", "exp3")
NB = json.load(open(os.path.join(PROJ_DIR, "exp3.ipynb"), encoding="utf-8"))

NUM = K.Numbering("3")
CJK = "微软雅黑"


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
    return K.add_eq(doc, os.path.join(ASSETS, "formula", key + ".png"), NUM)


def fig(doc, path, caption):
    return K.add_fig(doc, path, caption, NUM)


def msyh(doc, text, size, bold=False, align=None, indent_pt=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if indent_pt is not None:
        p.paragraph_format.first_line_indent = Pt(indent_pt)
    K.set_font(p.add_run(text), size=size, bold=bold, cn=CJK, ascii_font=CJK)
    return p


def doc_title(doc, text):
    p = msyh(doc, text, 18, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.first_line_indent = Pt(0)
    return p


def h1(doc, text):
    return msyh(doc, text, 14, bold=True)


def h2(doc, text):
    return msyh(doc, text, 12, bold=True)


def item(doc, text):
    return msyh(doc, text, 12, indent_pt=21)


def data_table(doc, headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        p = t.cell(0, j).paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        K.set_font(p.add_run(h), size=9.5, bold=True)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            K.set_font(p.add_run(str(v)), size=9.5)
    return t


def model_code(path, marker):
    text = open(path, encoding="utf-8").read()
    i = text.index(marker)
    return text[:i].rstrip(), text[i:].strip()


# ===========================================================================
def build_section(doc):
    doc_title(doc, "实验三 贝叶斯分类与BP算法")

    # ---------------------------------------------------------------- 一
    h1(doc, "一、贝叶斯分类器")
    h2(doc, "1.实验目标")
    item(doc, "掌握贝叶斯分类器的原理与应用方法。")
    h2(doc, "2、实验数据")
    item(doc, "过滤垃圾邮件数据(email datasets)：包含50条样本数据，每条数据为一个邮件，"
              "属于2个不同的类别。")
    h2(doc, "3、实验要求")
    for s in ["a.编写Python代码实验贝叶斯分类器。",
              "b.加载数据集后，进行基本的数据清洗和处理",
              "c.将数据集分为训练集和测试集，使用K折交叉验证来评估模型效果。",
              "d.训练贝叶斯分类器并对测试集进行预测。",
              "e.计算并报告模型的准确率、混淆矩阵、精确率、召回率和F1分数。",
              "f.绘制分类结果的混淆矩阵图。"]:
        item(doc, s)

    h2(doc, "4.实验步骤")
    K.body(doc, "本实验在 Python 3.12 环境下完成：scikit-learn 提供朴素贝叶斯、TF-IDF 向量化"
                "与 BP 网络的调包实现；自实现封装在 models/exp3/ 下——naive_bayes.py 实现"
                "多项式朴素贝叶斯，bp_network.py 用 numpy 实现单隐藏层 BP 网络（ReLU + Adam）。"
                "首先导入依赖库与自实现模型：")
    K.code_box(doc, code(1))

    # ---- a
    K.req_box(doc, "a.编写Python代码实验贝叶斯分类器。")
    K.label(doc, "【代码实现】")
    K.body(doc, "调包实现直接创建 sklearn 的 MultinomialNB：")
    K.code_box(doc, code(3))
    K.body(doc, "自实现的 MultinomialNB 封装在 models/exp3/naive_bayes.py 中：用 numpy 统计类"
                "先验与类条件概率，支持稀疏的 TF-IDF 矩阵，接口与 sklearn 一致：")
    K.code_box(doc, open(os.path.join(PROJ_DIR, "models", "exp3", "naive_bayes.py"),
                         encoding="utf-8").read())

    K.label(doc, "【原理与推导】")
    K.body(doc, "朴素贝叶斯分类器以贝叶斯公式为基础，把“已知邮件内容、判断它属于哪一类”"
                "转化为由先验与类条件概率计算后验概率的问题：")
    e1 = eq(doc, "f3_bayes")
    K.body(doc, f"式 {e1} 中 P(c) 为类别 c 的先验概率，P(x|c) 为该类别下出现这封邮件内容的"
                "概率，分母 P(x) 与类别无关，因此判决时只需比较分子，把邮件判给后验概率最大"
                "的类别：")
    e2 = eq(doc, "f3_argmax")
    K.body(doc, "先验概率直接用各类样本所占比例估计，类条件概率按词在类内的出现频次估计：")
    e3 = eq(doc, "f3_prior")
    e4 = eq(doc, "f3_mle")
    K.body(doc, f"式 {e3}、式 {e4} 构成训练阶段需要估计的全部参数。其中“朴素”指的是式 {e2} "
                "把邮件内容拆成各个词并假设它们在给定类别下相互独立，从而把联合概率写成连乘"
                "形式：")
    fig(doc, J("theory", "fig3_bayes_principle.jpg"), "贝叶斯决策的要素与“朴素”条件独立假设")
    K.body(doc, "该假设在现实中显然不完全成立（词与词之间存在搭配关系），但它使需要估计的"
                "参数量由指数级降为线性级。本实验的邮件样本只有 50 条，正是这一简化让模型在"
                "小样本下依然可用。")
    K.body(doc, f"式 {e4} 存在一个致命缺陷：若某个词在某一类的训练样本中从未出现，其条件"
                "概率会被估成 0，连乘之后整封邮件的后验概率都变为 0。为此引入拉普拉斯平滑，"
                "给每个词的计数加一个正数 α、分母加上 αV（V 为词表大小）：")
    e5 = eq(doc, "f3_laplace")
    K.body(doc, f"式 {e5} 保证任何词的条件概率都不为零。下图用示意数据说明零概率问题及其"
                "修正效果。")
    fig(doc, J("theory", "fig3_laplace_smoothing.jpg"), "零概率问题与拉普拉斯平滑")
    K.body(doc, "最后，为避免大量小数连乘导致数值下溢，实际计算在对数域进行，连乘变为求和：")
    e6 = eq(doc, "f3_logpost")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建了 MultinomialNB 对象（即多项式朴素贝叶斯，适用于以词频为"
                "特征的文本数据），尚未训练，因此没有输出。")

    # ---- b
    K.req_box(doc, "b.加载数据集后，进行基本的数据清洗和处理")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(5))
    K.body(doc, "读入后统一把文本转为小写，使大小写不同的同一个词被视为同一个特征：")
    K.code_box(doc, code(6))
    K.label(doc, "【原理与推导】")
    K.body(doc, "邮件以单个 txt 文件的形式存放在 ham 与 spam 两个目录下，共 50 个文件，"
                "两个类别各 25 封，样本数量完全均衡。读取时按目录来源把标签设为 0（正常邮件）"
                "与 1（垃圾邮件），并把文件内容与标签组装成 DataFrame。")
    K.body(doc, "清洗工作主要是统一大小写：英文文本中 “Free” 与 “free” 若被当作两个不同的"
                "特征，会稀释每个特征的统计量，在样本量很小的情况下影响尤为明显。转小写后"
                "词表规模减小、参数估计更稳定；标点等符号在向量化阶段会由分词规则过滤，"
                "停用词则在 TF-IDF 中通过 stop_words=\"english\" 去除。")
    K.body(doc, "两类邮件的长度分布非常接近，说明邮件长度本身不构成判别依据，模型必须从"
                "用词差异中学习。")
    fig(doc, J("theory", "fig3_email_overview.jpg"), "邮件数据集的类别分布与文本长度分布")
    K.label(doc, "【运行结果】")
    K.body(doc, "读取与清洗的单元格不产生输出；由构造结果可知共读入 50 封邮件，"
                "其中正常邮件 25 封、垃圾邮件 25 封。")

    # ---- c
    K.req_box(doc, "c.将数据集分为训练集和测试集，使用K折交叉验证来评估模型效果。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(8))
    K.code_box(doc, code(9))
    K.code_box(doc, code(10))
    K.code_box(doc, code(11))
    K.body(doc, "自实现：按同样的折划分逐折训练，每一折都只在训练部分上拟合 TF-IDF：")
    K.code_box(doc, code(12))
    K.label(doc, "【原理与推导】")
    K.body(doc, "文本无法直接输入模型，需要先向量化为数值特征。代码使用 TF-IDF 把每封邮件"
                "表示成词权向量，某个词在邮件中出现得越多、在整个语料中出现得越少，其权重"
                "越大：")
    e7 = eq(doc, "f3_tfidf")
    K.body(doc, f"式 {e7} 中 tf(t, d) 为词 t 在邮件 d 中的频次，df(t) 为包含该词的邮件数，"
                "N 为邮件总数。TF-IDF 权重非负，与多项式朴素贝叶斯的计数假设相容。")
    K.body(doc, "把向量化与分类器封装成 Pipeline，再按 8 : 2 分层划分训练集与测试集。使用 "
                "Pipeline 的关键作用是：交叉验证的每一折中，TF-IDF 只在训练部分上拟合，"
                "不会“看到”验证部分的词频统计，从而避免信息泄漏。K 折交叉验证以 K 个结果的"
                "均值作为性能估计：")
    e8 = eq(doc, "f3_cv")
    K.label(doc, "【运行结果】")
    K.body(doc, "5 折的准确率分别为 100%、100%、90%、80%、100%，平均 94.00%。折间波动达 "
                "20 个百分点，原因是样本总量仅 50 条，每折只有 10 封邮件，判错一封就会带来 "
                "10 个百分点的变化：")
    K.out_box(doc, out_text(11))
    fig(doc, J("theory", "fig3_cv_scores.jpg"), "5 折交叉验证的每折准确率与平均准确率")
    K.body(doc, "自实现逐折复现的结果与调包实现完全一致（每折、平均值均相同），说明两者的"
                "训练与预测逻辑等价：")
    K.out_box(doc, out_text(12))
    fig(doc, J("compare", "fig3_nb_compare.jpg"),
        "两种实现的交叉验证结果与类条件概率对比")

    # ---- d
    K.req_box(doc, "d.训练贝叶斯分类器并对测试集进行预测。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(14))
    K.code_box(doc, code(15))
    K.body(doc, "自实现：用同一份 TF-IDF 矩阵训练，并与调包实现的预测结果对比：")
    K.code_box(doc, code(20))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"训练阶段并不需要迭代优化，只是统计参数：按式 {e3} 估计先验、按式 {e5} "
                f"估计各词在两类下的条件概率；预测阶段则按式 {e6} 累加对数后验并取最大者。"
                "在对数域上可以看出，判决完全由各词的“对数似然比”共同决定，因此可以直接"
                "比较同一个词在两个类别下的权重差，把它作为该词判别能力的度量：")
    e9 = eq(doc, "f3_odds")
    K.body(doc, f"式 {e9} 取值越大说明该词越指向垃圾邮件，越小则越指向正常邮件。")
    fig(doc, J("theory", "fig3_top_words.jpg"),
        "对数似然比最大（判为垃圾邮件）与最小（判为正常邮件）的词")
    K.label(doc, "【运行结果】")
    K.body(doc, "两种实现在测试集上的准确率均为 90.00%，预测逐样本一致；类条件对数概率的"
                "最大差异为 0，说明自实现与 sklearn 的估计结果完全相同：")
    K.out_box(doc, out_text(20))

    # ---- e
    K.req_box(doc, "e.计算并报告模型的准确率、混淆矩阵、精确率、召回率和F1分数。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(17))
    K.code_box(doc, code(18))
    K.code_box(doc, code(19))
    K.label(doc, "【原理与推导】")
    K.body(doc, "准确率与误差率互补，是最直观的指标；但在文本分类中还需要混淆矩阵以及由它"
                "导出的查准率、查全率与 F1 分数：")
    e10 = eq(doc, "f3_error")
    e11 = eq(doc, "f3_prf")
    K.body(doc, f"式 {e11} 中，就垃圾邮件识别而言，查准率反映“被判为垃圾邮件的邮件中有多少"
                "真的是垃圾邮件”（是否误伤正常邮件），查全率反映“真正的垃圾邮件被拦截了"
                "多少”（是否漏判），F1 则是二者的调和平均。")
    K.body(doc, "需要说明的是，本实验测试集只有 10 封邮件（正常邮件与垃圾邮件各 5 封），"
                "判错任何一封都会使准确率下降 10 个百分点，因此单次测试准确率应结合 5 折"
                "交叉验证的平均准确率一起解读。")
    K.label(doc, "【运行结果】")
    K.body(doc, "测试集准确率为 90.00%：")
    K.out_box(doc, out_text(17))
    K.body(doc, "混淆矩阵显示：5 封正常邮件全部判对；5 封垃圾邮件中 4 封判对、1 封被漏判"
                "为正常邮件：")
    K.out_box(doc, out_text(18))
    K.body(doc, "分类报告给出每一类的查准率、查全率与 F1 分数：垃圾邮件的查准率为 1.00（没有"
                "误伤正常邮件），查全率为 0.80（漏掉了一封），正常邮件的查准率为 0.83、查全率"
                "为 1.00：")
    K.out_box(doc, out_text(19))

    # ---- f
    K.req_box(doc, "f.绘制分类结果的混淆矩阵图。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(22))
    K.body(doc, "为便于对比，在同一张图中并排画出两种实现的混淆矩阵：")
    K.code_box(doc, code(23))
    K.label(doc, "【原理与推导】")
    K.body(doc, "混淆矩阵图把四个计数放进同一张热力图：对角线颜色越深说明判对的样本越多，"
                "非对角线则对应把正常邮件误判为垃圾邮件、或把垃圾邮件漏判这两类错误。"
                "在垃圾邮件过滤这类应用中，两类错误的代价并不相同——漏判只是多收到一封"
                "骚扰邮件，误判则可能造成重要邮件丢失，因此实际部署时往往需要结合代价"
                "调整判决阈值，而不能只看准确率。")
    K.label(doc, "【运行结果】")
    fig(doc, J("compare", "fig3_nb_cm_compare.jpg"), "两种实现的混淆矩阵对比")

    # ---------------------------------------------------------------- 二
    doc.add_page_break()
    h1(doc, "二、BP算法")
    h2(doc, "1.实验目标")
    item(doc, "掌握BP算法的原理及其应用。")
    h2(doc, "2. 实验数据")
    item(doc, "考虑UCI数据库中关于红酒品质的数据集(Win Quality Data Set)。该数据集包含1599个"
              "红葡萄酒样本信息，记录了非挥发性酸含量、挥发性酸含量、柠檬酸含量等11个特征，"
              "响应变量为综合品质得分，取值0~10，此样本集仅包含3~8这6种评分。")
    h2(doc, "3. 实验要求")
    for s in ["a.读入数据；",
              "b.将红酒分为两个等级，其中评分3、4、5的为”bad”品质，评分为6、7、8的分为"
              "“good”品质。",
              "c. 将特征变量进行归一化处理。",
              "d. 设置一个随机种子，将数据集按7：3的比例划分为训练集和测试集。",
              "e.使用BP算法在训练数据集上构建合适的神经网络模型，隐藏层神经元个数设置为2.",
              "f.在测试集上计算模型的测试误差率。",
              "g.选取不同的隐藏层神经元个数构建神经网络模型，根据不同模型的测试错误率"
              "选出最优模型。"]:
        item(doc, s)

    h2(doc, "4.实验步骤")
    K.req_box(doc, "a.读入数据；")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(26))
    K.label(doc, "【原理与推导】")
    K.body(doc, "红酒品质数据集共 1599 条样本，包含 11 个理化指标特征（非挥发性酸、挥发性酸、"
                "柠檬酸、残糖、氯化物、游离二氧化硫、总二氧化硫、密度、pH、硫酸盐、酒精）"
                "与 1 个品质得分 quality，数据完整、没有缺失值。quality 的实际取值为 3~8 分，"
                "其中 5 分（681 条）和 6 分（638 条）占绝大多数，3 分与 8 分分别只有 10 条和 "
                "18 条，分布明显集中。")
    K.label(doc, "【运行结果】")
    K.out_box(doc, out_text(26))
    fig(doc, J("theory", "fig3_quality_distribution.jpg"), "红酒综合品质分布与二分类划分")

    K.req_box(doc, "b.将红酒分为两个等级，其中评分3、4、5的为”bad”品质，评分为6、7、8的"
                   "分为“good”品质。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(28))
    K.label(doc, "【原理与推导】")
    K.body(doc, "把 6 档评分合并为二分类后，bad 类（3~5 分）共 744 条、占 46.5%，good 类"
                "（6~8 分）共 855 条、占 53.5%，两类比例接近，属于较均衡的二分类问题。")
    K.body(doc, "需要指出的是，把有序的品质评分强行切成两类会损失一部分信息：5 分（bad 类"
                "上限）与 6 分（good 类下限）在口感上差异很小，位于分界附近的样本本身就"
                "难以区分，这构成了本题误差率的下限。")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只新增 grade 与 label 两列，没有输出。")

    K.req_box(doc, "c. 将特征变量进行归一化处理。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(30))
    K.code_box(doc, code(31))
    K.label(doc, "【原理与推导】")
    K.body(doc, "11 个特征的量纲差异极大：总二氧化硫的取值可达数百，氯离子含量只有 0.01 量级，"
                "密度在 0.99 附近。神经网络使用梯度下降训练，若不做处理，参数更新会被量纲大的"
                "特征主导。MinMaxScaler 把每个特征线性映射到 [0, 1]：")
    e12 = eq(doc, "f3_minmax")
    K.body(doc, f"式 {e12} 只依赖特征的极值，不改变样本之间的相对次序。")
    fig(doc, J("theory", "fig3_scaling_effect.jpg"), "归一化前后各特征取值的分布对比")
    K.label(doc, "【运行结果】")
    K.body(doc, "两个单元格都不产生输出。")

    K.req_box(doc, "d. 设置一个随机种子，将数据集按7：3的比例划分为训练集和测试集。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(33))
    K.code_box(doc, code(34))
    K.label(doc, "【原理与推导】")
    K.body(doc, "代码设置 random_state = 42，使每次运行得到完全相同的划分，保证结果可复现；"
                "同时启用 stratify 做分层采样，使训练集与测试集中 bad/good 的比例与整体一致：")
    e13 = eq(doc, "f3_split")
    K.body(doc, f"如式 {e13} 所示，训练集用于拟合网络参数，测试集只在评估阶段使用。")
    K.label(doc, "【运行结果】")
    K.body(doc, "两个单元格都不产生输出；由样本总数可推知划分结果为训练集 1119 条、"
                "测试集 480 条。")

    K.req_box(doc, "e.使用BP算法在训练数据集上构建合适的神经网络模型，隐藏层神经元个数"
                   "设置为2.")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(36))
    K.body(doc, "自实现的 BP 网络封装在 models/exp3/bp_network.py 中：用 numpy 实现单隐藏层"
                "网络、ReLU 激活、交叉熵损失、反向传播与 Adam 优化，接口与 sklearn 的 "
                "MLPClassifier 一致。文件前半部分为工具函数与参数初始化：")
    bp1, bp2 = model_code(os.path.join(PROJ_DIR, "models", "exp3", "bp_network.py"),
                          "    # ------------------------------------------------------------------\n"
                          "    def _init_params")
    K.code_box(doc, bp1)
    K.body(doc, "文件后半部分为前向/反向传播、Adam 更新与预测接口：")
    K.code_box(doc, bp2)

    K.label(doc, "【原理与推导】")
    K.body(doc, "神经网络的基本单元是神经元：它对输入特征做加权求和，再经激活函数得到输出：")
    e14 = eq(doc, "f3_neuron")
    K.body(doc, f"式 {e14} 中的激活函数 f 赋予网络非线性能力。本实验使用 ReLU：")
    e15 = eq(doc, "f3_relu")
    K.body(doc, f"式 {e15} 在正区间导数为常数 1，能有效缓解 Sigmoid 类函数在饱和区梯度趋近"
                "于 0 的问题；作为对比，Sigmoid 的表达式为：")
    e16 = eq(doc, "f3_sigmoid")
    fig(doc, J("theory", "fig3_neuron_activation.jpg"), "神经元结构与常用激活函数")
    K.body(doc, "BP 算法包含两个过程。前向传播逐层计算各层输出，直到得到网络的预测值：")
    e17 = eq(doc, "f3_forward")
    K.body(doc, f"随后用交叉熵衡量预测与真实标签之间的偏差：")
    e18 = eq(doc, "f3_loss")
    K.body(doc, "反向传播则利用链式法则把误差从输出层逐层向前传递，先求输出层的误差项：")
    e19 = eq(doc, "f3_delta_out")
    K.body(doc, f"再由后一层的误差项递推得到各隐藏层的误差项：")
    e20 = eq(doc, "f3_delta_hidden")
    K.body(doc, f"式 {e19}、式 {e20} 是 BP 算法的核心。得到各层误差项后，参数的梯度可以写成"
                "误差项与本层输入的乘积，并按梯度下降更新：")
    e21 = eq(doc, "f3_grad")
    e22 = eq(doc, "f3_update")
    fig(doc, J("theory", "fig3_bp_flow.jpg"), "BP 网络的前向传播与误差反向传播")
    K.label(doc, "【运行结果】")
    K.out_box(doc, out_text(36))

    K.req_box(doc, "f.在测试集上计算模型的测试误差率。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(38))
    K.code_box(doc, code(39))
    K.body(doc, "自实现：用同样的结构（隐藏层 2 个神经元）训练并计算误差率：")
    K.code_box(doc, code(40))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"测试误差率与准确率互补，即式 {e10}。误差率越低说明模型的泛化能力越好。")
    K.body(doc, "下图给出隐藏层为 2、1、10 个神经元时的训练损失下降曲线。可以看出损失都在"
                "前几十轮迅速下降，随后趋于平缓；但损失下降得低并不等于测试效果好，网络仍"
                "可能把训练集“背下来”而在测试集上失效。")
    fig(doc, J("theory", "fig3_loss_curves.jpg"), "不同隐藏层规模的训练损失下降过程")
    K.label(doc, "【运行结果】")
    K.body(doc, "调包实现的测试误差率为 54.17%，即准确率仅 45.83%，甚至低于“全部预测为 "
                "good”的 46.5% 误差率；自实现在同样结构下的误差率为 25.62%：")
    K.out_box(doc, out_text(40))
    K.body(doc, "从混淆矩阵可以看清调包实现退化的原因：该网络把所有测试样本都判为 bad，"
                "257 个 good 类样本无一被识别出来，说明隐藏层的 2 个 ReLU 单元在该次训练中"
                "几乎全部失活，网络退化成了常数分类器。")
    fig(doc, J("concept", "fig3_bp_confusion.jpg"),
        "调包实现（隐藏层 2 个神经元）在测试集上的混淆矩阵")

    K.req_box(doc, "g.选取不同的隐藏层神经元个数构建神经网络模型，根据不同模型的测试错误率"
                   "选出最优模型。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(42))
    K.code_box(doc, code(43))
    K.body(doc, "自实现：遍历同样的六种规模，并与调包实现逐项对比：")
    K.code_box(doc, code(44))
    K.label(doc, "【原理与推导】")
    K.body(doc, "隐藏层神经元个数决定了网络的表示能力：单元太少会欠拟合，太多则容易过拟合，"
                "同时非凸优化也更容易落到较差的局部最优。为此对 1、2、3、5、10、15 六种规模"
                "分别训练并比较测试误差率。")
    K.body(doc, "两种实现的对比给出了很有意思的结果：除 2 个神经元外，两套实现的误差率都稳定"
                "在 26.0%~26.9% 之间，说明此时限制模型效果的主要因素并不是网络容量，而是 11 个"
                "理化指标与品质等级之间本身就存在的重叠。")
    K.body(doc, "2 个神经元的差异则非常明显：调包实现的误差率高达 54.17%，原因是两个 ReLU "
                "单元同时失活、网络退化为常数分类器；自实现在同一结构下正常收敛到 25.62%。"
                "这说明该结构正处于表示能力的临界点上，结果高度依赖参数初始化与优化路径，"
                "也印证了非凸损失曲面上不同初值会收敛到不同局部最优。")
    K.body(doc, "若以测试误差率作为选择标准：调包实现的最优模型是隐藏层 1 个神经元"
                "（26.04%），自实现的最优模型是隐藏层 2 个神经元（25.62%）。但考虑到测试集"
                "只有 480 条样本，26.04% 与 25.62% 之间不到 2 个样本的差异属于随机波动，"
                "更稳妥的结论是：3 个以上神经元都能取得相近效果，2 个神经元是需要谨慎对待的"
                "配置。")
    fig(doc, J("compare", "fig3_bp_compare.jpg"), "两种 BP 实现在不同隐藏层规模下的误差率对比")
    K.label(doc, "【运行结果】")
    K.body(doc, "六种规模的对比结果如下：")
    K.out_box(doc, out_text(44))
    K.body(doc, "按测试误差率最小选出的最优模型为：")
    K.out_box(doc, out_text(43))

    doc.add_page_break()


def main():
    backup = "/tmp/总实验报告_Copy_实验二后备份.docx"
    shutil.copy(COPY, backup)
    print("已备份 ->", backup)

    doc = Document(COPY)
    body = doc.element.body

    stop = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("实验四"):
            stop = p._element
            break
    start = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("实验三"):
            start = p._element
            break
    if stop is None or start is None:
        raise RuntimeError("未找到实验三或实验四标题")

    cover = None
    el = start
    while True:
        el = el.getnext()
        if el is None or el is stop:
            break
        if el.tag == qn("w:tbl"):
            cover = el
            break

    sect = body.find(qn("w:sectPr"))
    start_len = list(body).index(sect) if sect is not None else len(list(body))
    build_section(doc)
    new_els = [el for el in list(body)[start_len:] if el.tag != qn("w:sectPr")]
    print(f"新章节元素数：{len(new_els)}")

    removed = 0
    el = start
    while el is not None and el is not stop:
        nxt = el.getnext()
        if el is not cover:
            body.remove(el)
            removed += 1
        el = nxt
    print(f"已删除旧章节元素：{removed}")

    for el in new_els:
        stop.addprevious(el)
    if cover is not None:
        new_els[0].addnext(cover)

    doc.save(COPY)
    print("已写入 ->", COPY)
    print(f"插图 {NUM.fig} 张，公式 {NUM.eq} 个")


if __name__ == "__main__":
    main()
