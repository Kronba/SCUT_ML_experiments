# -*- coding: utf-8 -*-
"""实验四：调包实现 + 自实现 的新版报告，写入 总实验报告 (Copy).docx。

替换原实验四章节（文档最后一节）。
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
ASSETS = os.path.join(REPORT_DIR, "素材", "exp4")
NB = json.load(open(os.path.join(PROJ_DIR, "exp4.ipynb"), encoding="utf-8"))

NUM = K.Numbering("4")
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
    doc_title(doc, "实验四 集成模型与决策树")

    # ---------------------------------------------------------------- 一
    h1(doc, "一、AdaBoost集成学习")
    h2(doc, "1.实验目标")
    item(doc, "a.理解集成学习的基本思想，掌握AdaBoost算法的原理。")
    item(doc, "b.学会使用AdaBoost算法解决实际问题，并评估其性能。")
    h2(doc, "2.实验数据")
    item(doc, "美国某银行信用卡申请数据：包含1548条记录，含18个特征，其中Ind_ID为唯一标识，"
              "label为标签（0表示未违约、1表示违约）。")
    h2(doc, "3.实验要求")
    for s in ["a.编写Python代码实现AdaBoost模型，选择一个基本分类器作为弱学习器。",
              "b.读取数据集，处理缺失值（如有），并将数据集分为训练集和测试集。",
              "c.在训练集上训练AdaBoost模型。在测试集上进行预测，并计算准确率。",
              "d.打印混淆矩阵，打印分类报告，显示每个类别的精确率、召回率和F1分数。",
              "e.绘制混淆均值的热力图。"]:
        item(doc, s)

    h2(doc, "4.实验步骤")
    K.body(doc, "本实验在 Python 3.12 环境下完成：scikit-learn 提供 AdaBoost、决策树与数据"
                "预处理的调包实现；自实现封装在 models/exp4/ 下——adaboost.py 用 numpy 实现"
                "决策树桩与 AdaBoost 的加权投票，decision_tree.py 实现带类别权重的 CART 决策树，"
                "接口均与 scikit-learn 一致。首先导入依赖库与自实现模型：")
    K.code_box(doc, code(1))

    # ---- a
    K.req_box(doc, "a.编写Python代码实现AdaBoost模型，选择一个基本分类器作为弱学习器。")
    K.label(doc, "【代码实现】")
    K.body(doc, "调包实现以深度为 1 的决策树（决策树桩）作为弱学习器：")
    K.code_box(doc, code(3))
    K.body(doc, "自实现封装在 models/exp4/adaboost.py 中。文件前半部分是决策树桩：在每个特征上"
                "枚举阈值，按加权错误率选出最优的“特征—阈值—左右类别”组合：")
    ada1, ada2 = model_code(os.path.join(PROJ_DIR, "models", "exp4", "adaboost.py"),
                            "class AdaBoostClassifier")
    K.code_box(doc, ada1)
    K.body(doc, "文件后半部分是 AdaBoost 主体：逐轮训练弱学习器、计算其权重 α、按指数损失"
                "调整样本权重：")
    K.code_box(doc, ada2)

    K.label(doc, "【原理与推导】")
    K.body(doc, "AdaBoost 的核心思想是“三个臭皮匠顶个诸葛亮”：把多个弱学习器按各自的权重"
                "组合起来，得到比任何一个成员都强的强分类器：")
    e1 = eq(doc, "f4_ensemble")
    K.body(doc, f"式 {e1} 中 h_t 为第 t 个弱学习器，α_t 为其权重。本实验的弱学习器是决策树桩，"
                "它只在某个特征上取一个阈值，把样本分成左右两部分并各给一个类别：")
    e2 = eq(doc, "f4_stump")
    K.body(doc, f"单棵树桩的表达能力很弱，但它计算极快，而且只要样本不是完全随机，加权错误率"
                "总能低于 0.5，从而满足 AdaBoost 对弱学习器的要求。")
    K.body(doc, "训练过程是串行的：每一轮先计算当前弱学习器在加权样本上的错误率：")
    e3 = eq(doc, "f4_eps")
    K.body(doc, "再据此计算该弱学习器的权重：")
    e4 = eq(doc, "f4_alpha")
    K.body(doc, f"式 {e4} 说明错误率越低，α 越大，该学习器在最终投票中越有话语权；当错误率"
                "等于 0.5 时 α = 0，该学习器不参与投票。随后按指数损失更新样本权重：")
    e5 = eq(doc, "f4_wupdate")
    K.body(doc, f"式 {e5} 使被分错的样本权重变大、分对的变小，于是下一轮的弱学习器会“专注”"
                "于之前做错的样本。归一化因子 Z_t 保证权重之和为 1：")
    e6 = eq(doc, "f4_Z")
    fig(doc, J("theory", "fig4_adaboost_principle.jpg"),
        "AdaBoost 的串行训练流程：更新样本权重与弱学习器权重")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建了弱学习器对象，尚未训练，因此没有输出。")

    # ---- b
    K.req_box(doc, "b.读取数据集，处理缺失值（如有），并将数据集分为训练集和测试集。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(5))
    K.code_box(doc, code(6))
    K.code_box(doc, code(7))
    K.code_box(doc, code(8))
    K.code_box(doc, code(9))
    K.label(doc, "【原理与推导】")
    K.body(doc, "数据由特征表与标签表两个文件组成，通过 Ind_ID 内连接合并。合并后共 1548 条"
                "样本、18 个特征，其中 Payment_Value 等字段存在缺失值，需要先做缺失值处理：")
    e7 = eq(doc, "f4_impute")
    K.body(doc, f"式 {e7} 表示数值特征用中位数填补、类别特征用众数填补；中位数比均值更稳健，"
                "不会因为个别极端值而偏移。")
    K.body(doc, "特征中既有数值型（如年收入、年龄），也有类别型（如职业、住房类型），而"
                "决策树桩只能比较大小，因此需要对类别特征做独热编码：")
    e8 = eq(doc, "f4_onehot")
    K.body(doc, f"式 {e8} 把每个类别映射成一个 0/1 指示向量。上述缺失值填补与独热编码被封装"
                "在 ColumnTransformer 中，训练时只对训练集拟合、再套用到测试集，避免信息泄漏。")
    fig(doc, J("theory", "fig4_pipeline.jpg"), "数据预处理与建模流程")
    K.body(doc, "数据按 7 : 3 分层划分为训练集（1083 条）与测试集（465 条），比例与整体一致：")
    e9 = eq(doc, "f4_ratio")
    K.body(doc, "该数据集是典型的类别不平衡问题：违约样本仅占 11.4%，若模型不加学习、一律预测"
                "为“未违约”，准确率也能达到：")
    e10 = eq(doc, "f4_baseline")
    fig(doc, J("theory", "fig4_imbalance_baseline.jpg"),
        "信用卡数据的类别不平衡与“全判为多数类”的基线准确率")
    K.label(doc, "【运行结果】")
    K.body(doc, "合并与预处理后，特征维度由 18 列扩展到 53 维（独热编码所致），训练集 1083 条、"
                "测试集 465 条。")

    # ---- c
    K.req_box(doc, "c.在训练集上训练AdaBoost模型。在测试集上进行预测，并计算准确率。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(11))
    K.code_box(doc, code(12))
    K.code_box(doc, code(13))
    K.code_box(doc, code(14))
    K.body(doc, "自实现：复用同一套预处理，训练自实现的 AdaBoost 并与调包实现对比：")
    K.code_box(doc, code(18))
    K.label(doc, "【原理与推导】")
    K.body(doc, "代码使用 100 个决策树桩、学习率 1.0。由于类别极不平衡，弱学习器会很快把"
                "多数类学好，随后几轮主要纠正少数类上的错误；但 100 个树桩的加权投票仍不足以"
                "克服“违约样本太少”这一根本困难，这一点在下文的混淆矩阵中表现得很清楚。")
    K.body(doc, "自实现与调包实现优化的是同一个目标：两者都使用加权错误率选桩、用同一个 α "
                "公式赋权、用同一条指数损失更新样本权重，因此结果应当非常接近。")
    K.label(doc, "【运行结果】")
    K.body(doc, "调包实现的测试准确率为 88.17%，自实现为 87.96%，两者仅差 2 个样本：")
    data_table(doc, ["实现方式", "准确率", "弱学习器数", "预测一致率"],
               [["sklearn 调包", "88.17%", "100", "—"],
                ["自实现", "87.96%", "100", "与调包 99.78% 一致"]])
    K.body(doc, "准确率 88.17% 看似不低，但常数基线“全判为未违约”就有 88.60%，也就是说"
                "模型实际上没有超过基线——原因在下一节的混淆矩阵中可以看到。")
    K.out_box(doc, out_text(18))

    # ---- d
    K.req_box(doc, "d.打印混淆矩阵，打印分类报告，显示每个类别的精确率、召回率和F1分数。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(16))
    K.code_box(doc, code(17))
    K.label(doc, "【原理与推导】")
    K.body(doc, "准确率的定义以及由混淆矩阵导出的查准率、查全率与 F1 分数为：")
    e11 = eq(doc, "f4_acc")
    e12 = eq(doc, "f4_prf")
    K.body(doc, f"就信用卡违约识别而言，把“违约”视为正类：查准率是“被判为违约的申请中真正违约"
                "的比例”，查全率是“真正违约的申请中被识别出来的比例”。两者往往此消彼长，因此"
                "需要 F1 综合衡量：")
    e13 = eq(doc, "f4_macro")
    K.body(doc, f"式 {e13} 的宏平均对每一类先算指标再取平均，能反映少数类的表现，而不是被"
                "多数类“淹没”。")
    K.label(doc, "【运行结果】")
    K.body(doc, "混淆矩阵与分类报告如下：")
    K.out_box(doc, out_text(16))
    K.out_box(doc, out_text(17))
    K.body(doc, "从结果看：412 个未违约样本中 410 个判对、2 个误判；53 个违约样本全部被漏判。"
                "也就是说，模型把几乎所有样本都判成了“未违约”，准确率高只是因为多数类占比高，"
                "对少数类毫无识别能力——这正是类别不平衡问题中最典型的失败模式。")
    fig(doc, J("concept", "fig4_metrics_bars.jpg"),
        "AdaBoost 的分类报告指标与常数基线对比")

    # ---- e
    K.req_box(doc, "e.绘制混淆均值的热力图。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(20))
    K.body(doc, "为便于对比，在同一张图中并排画出两种实现的混淆矩阵：")
    K.code_box(doc, code(21))
    K.label(doc, "【原理与推导】")
    K.body(doc, "热力图用颜色深浅表示混淆矩阵中的计数，对角线是判对的样本，非对角线是对应的"
                "两类错误。把两套实现并排放置，可以直观判断两者是否犯了同样的错误。")
    K.label(doc, "【运行结果】")
    fig(doc, J("notebook_out", "cell21_0.jpg"), "两种 AdaBoost 实现的混淆矩阵对比")
    K.body(doc, "两张热力图几乎完全相同：两者都把 1~3 个未违约样本误判为违约，且都没有识别出"
                "任何一个违约样本，说明这一失败模式来自数据本身的不平衡，而不是实现细节。")

    # ---------------------------------------------------------------- 二
    doc.add_page_break()
    h1(doc, "二、决策树（非均衡分类问题）")
    h2(doc, "1.实验目标")
    item(doc, "a.理解决策树算法的原理。")
    item(doc, "b.学会使用决策树算法解决实际问题。")
    item(doc, "c.学会处理非均衡分类问题。")
    item(doc, "d.学会对决策树模型进行可视化和解释。")
    h2(doc, "2.实验数据")
    item(doc, "Florence 数据集：包含 4000 条样本，记录了某品牌皮具的购买记录，特征包括性别、"
              "年龄、收入、是否有孩子、是否购买等。其中 Florence 为标签，1 表示购买、0 表示"
              "未购买。")
    h2(doc, "3.实验要求")
    for s in ["a.编写Python代码实现决策树模型。（考虑平衡因子）",
              "b.将数据集分为训练集和测试集，构建决策树模型并进行训练。",
              "c.使用测试集评估模型的性能，包括准确率、混淆矩阵、每个类别的精确率、召回率和"
              "F1分数等评估指标。",
              "d.随机选取3个特征，重新训练并可视化决策树。"]:
        item(doc, s)

    h2(doc, "4.实验步骤")
    K.req_box(doc, "a.编写Python代码实现决策树模型。（考虑平衡因子）")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(25))
    K.body(doc, "自实现封装在 models/exp4/decision_tree.py 中：用 numpy 实现 CART 决策树，"
                "支持熵/基尼准则、深度限制、叶节点最小样本数与类别权重。文件前半部分是树节点"
                "结构与最优划分的搜索：")
    dt1, dt2 = model_code(os.path.join(PROJ_DIR, "models", "exp4", "decision_tree.py"),
                          "    # ------------------------------------------------------------------\n"
                          "    def fit(self, X, y, sample_weight=None):")
    K.code_box(doc, dt1)
    K.body(doc, "文件后半部分是建树、预测与特征重要性统计：")
    K.code_box(doc, dt2)

    K.label(doc, "【原理与推导】")
    K.body(doc, "决策树通过一系列“是/否”判断把样本逐步划分到叶节点，每个叶节点的预测值由"
                "落入其中的样本决定：")
    e14 = eq(doc, "f4_leaf")
    K.body(doc, f"式 {e14} 中 N_t 为叶节点样本数，N_tk 为其中第 k 类的样本数。划分的好坏由"
                "不纯度衡量，代码默认使用信息熵：")
    e15 = eq(doc, "f4_entropy")
    K.body(doc, "划分后按加权不纯度下降（信息增益）选择最优划分：")
    e16 = eq(doc, "f4_gini")
    K.body(doc, f"式 {e16} 中 I 为不纯度函数，N_L、N_R 为左右子节点的样本数。下图给出熵、"
                "基尼指数随正类比例变化的曲线，两者都在 0.5 处达到最大。")
    K.body(doc, "每次划分都选信息增益最大的“特征—阈值”组合：")
    e17 = eq(doc, "f4_infogain")
    K.body(doc, f"式 {e17} 中 H(D) 为划分前的不纯度。自实现即按此逐节点贪心搜索最优划分。")
    K.body(doc, "若不限制树的生长，决策树会把训练集完全分开（每个叶节点只有一个样本），"
                "这正是它容易过拟合的原因，因此需要用深度、叶节点样本数等参数剪枝：")
    e18 = eq(doc, "f4_prune")
    K.body(doc, f"式 {e18} 是预剪枝的约束：只有满足这些条件才允许继续划分。本实验设置"
                " max_depth = 5、min_samples_split = 10、min_samples_leaf = 5。")
    K.body(doc, "Florence 数据中“购买”样本只占 27.4%，属于非均衡分类问题。类别权重通过"
                "给少数类样本加权来平衡两类的影响：")
    e19 = eq(doc, "f4_classweight")
    K.body(doc, f"式 {e19} 中 N 为样本总数、K 为类别数、N_k 为第 k 类样本数。class_weight="
                "balanced 即按此自动赋权，使少数类在划分时不被忽视。")
    fig(doc, J("theory", "fig4_tree_principle.jpg"),
        "决策树的划分准则：熵、基尼指数与信息增益")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建决策树对象，尚未训练，因此没有输出。")

    K.req_box(doc, "b.将数据集分为训练集和测试集，构建决策树模型并进行训练。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(23))
    K.code_box(doc, code(27))
    K.code_box(doc, code(28))
    K.label(doc, "【原理与推导】")
    K.body(doc, "Florence 数据集共 4000 条样本、4 个特征（Gender、M、R、F 分别代表性别、"
                "购买金额、最近一次购买距今时间、购买频次），标签 Florence 表示是否购买，"
                "正类占 27.4%。按 8 : 2 分层划分后，训练集 3200 条、测试集 800 条。")
    fig(doc, J("theory", "fig4_florence_overview.jpg"),
        "Florence 数据集的类别分布与各特征取值分布")
    K.body(doc, "由于类别不平衡，若不加权，树会倾向于把样本判为多数类。启用 class_weight="
                "balanced 后，两类在划分时按样本数反比加权，少数类的影响被放大，从而在查全率"
                "与查准率之间取得更合理的折中：")
    fig(doc, J("concept", "fig4_balanced_compare.jpg"),
        "是否启用类别权重对两种类别指标的影响")
    K.label(doc, "【运行结果】")
    K.body(doc, "训练完成的模型对象如下：")
    K.out_box(doc, out_text(28))

    K.req_box(doc, "c.使用测试集评估模型的性能，包括准确率、混淆矩阵、每个类别的精确率、"
                   "召回率和F1分数等评估指标。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(30))
    K.code_box(doc, code(31))
    K.code_box(doc, code(32))
    K.code_box(doc, code(33))
    K.body(doc, "自实现：用同样的超参数训练自实现的决策树，并与调包实现对比：")
    K.code_box(doc, code(34))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"评价指标与 AdaBoost 部分相同：准确率（式 {e11}）、混淆矩阵、查准率、查全率"
                f"与 F1（式 {e12}、式 {e13}）。由于两类样本比例为 3 : 1，准确率 70% 左右"
                "虽然不高，但若正类的查全率与 F1 明显高于“全判为多数类”的做法，就说明模型"
                "确实学到了信息。")
    K.label(doc, "【运行结果】")
    K.body(doc, "调包实现的测试准确率为 55.63%，自实现为 56.25%，两者的预测一致率为 "
                "99.38%：")
    data_table(doc, ["实现方式", "准确率", "预测一致率"],
               [["sklearn 调包", "55.63%", "—"],
                ["自实现", "56.25%", "与调包 99.38% 一致"]])
    K.out_box(doc, out_text(33))
    fig(doc, J("notebook_out", "cell35_0.jpg"), "两种决策树实现的混淆矩阵对比")
    K.body(doc, "两套实现的混淆矩阵几乎一致：购买类（少数类）的召回率约 56%、未购买类的召回率"
                "约 56%，说明模型在两类之间做了折中，而不是简单地把所有样本判为多数类。")
    K.body(doc, "特征重要性方面，两种实现都认为 Gender 最不重要；但 M、F、R 三者的排序略有"
                "差异（调包实现把 M 排在首位，自实现把 F 排在首位）。原因在于两者的划分"
                "搜索在增益相近的候选上做了不同的取舍，树结构因此不完全相同，但预测结果"
                "几乎一致：")
    fig(doc, J("notebook_out", "cell35_2.jpg"), "两种决策树的特征重要性对比")

    K.req_box(doc, "d.随机选取3个特征，重新训练并可视化决策树。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(38))
    K.code_box(doc, code(39))
    K.code_box(doc, code(40))
    K.code_box(doc, code(41))
    K.code_box(doc, code(42))
    K.label(doc, "【原理与推导】")
    K.body(doc, "随机选取 3 个特征后重新训练，相当于在更小的特征子空间上学习。特征变少时"
                "单棵树的表达能力下降，但可视化结果更容易阅读；这也有助于理解每个特征对"
                "判断“是否购买”的具体作用。")
    K.body(doc, "用 3 个特征训练出的树见下图：")
    fig(doc, J("theory", "fig4_tree_3feat.jpg"),
        "仅用 3 个随机特征训练出的决策树（深度 5）")
    K.label(doc, "【运行结果】")
    K.body(doc, "随机抽到的 3 个特征与重新训练后的准确率如下：")
    K.out_box(doc, out_text(38))
    K.out_box(doc, out_text(42))
    K.body(doc, "与使用全部 4 个特征的 55.63% 相比，3 特征模型的准确率变化不大，说明第四个"
                "特征（Gender）对判别的贡献确实很小——这与特征重要性的结论相互印证。")
    K.body(doc, "完整的决策树如下图（深度 5，共 32 个叶节点）。图中每个节点标注了划分条件、"
                "熵值、样本数与各类比例：")
    fig(doc, J("theory", "fig4_tree_viz.jpg"), "完整决策树（Florence 数据集，深度 5）")
    K.body(doc, "从树的结构可以看出：M（购买金额）与 F（购买频次）在靠近根节点的位置被优先"
                "使用，说明它们对区分“是否购买”最有效；Gender 只出现在树的较深处，且划分"
                "带来的增益有限，与特征重要性排序一致。")
    fig(doc, J("theory", "fig4_importance.jpg"), "决策树的特征重要性排序")


def main():
    backup = "/tmp/总实验报告_Copy_实验三后备份.docx"
    shutil.copy(COPY, backup)
    print("已备份 ->", backup)

    doc = Document(COPY)
    body = doc.element.body

    start = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("实验四"):
            start = p._element
            break
    if start is None:
        raise RuntimeError("未找到实验四标题")

    cover = None
    el = start
    while True:
        el = el.getnext()
        if el is None or el.tag == qn("w:sectPr"):
            break
        if el.tag == qn("w:tbl"):
            cover = el
            break

    sect = body.find(qn("w:sectPr"))
    start_len = list(body).index(sect) if sect is not None else len(list(body))
    build_section(doc)
    new_els = [el for el in list(body)[start_len:] if el.tag != qn("w:sectPr")]
    print(f"新章节元素数：{len(new_els)}")

    # 删除旧章节：从实验四标题到 sectPr 之前
    removed = 0
    el = start
    while el is not None and el is not sect:
        nxt = el.getnext()
        if el is not cover:
            body.remove(el)
            removed += 1
        el = nxt
    print(f"已删除旧章节元素：{removed}")

    for el in new_els:
        sect.addprevious(el)
    if cover is not None:
        new_els[0].addnext(cover)

    doc.save(COPY)
    print("已写入 ->", COPY)
    print(f"插图 {NUM.fig} 张，公式 {NUM.eq} 个")


if __name__ == "__main__":
    main()
