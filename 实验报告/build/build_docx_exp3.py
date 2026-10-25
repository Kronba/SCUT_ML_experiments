# -*- coding: utf-8 -*-
"""实验三实验报告生成脚本（贝叶斯分类与BP算法）。

风格与实验一/二一致：只写"4.实验步骤"，每项要求按
【实验要求】框 → 【代码实现】代码框 → 【原理与推导】 → 【运行结果】组织；
不写 a/b/c/d 小标题；图注不带来源/主题后缀；代码直接从 exp3.ipynb 读取。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docx_kit as K

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
ASSETS = os.path.join(REPORT_DIR, "素材", "exp3")
NB = json.load(open(os.path.join(PROJ_DIR, "exp3.ipynb"), encoding="utf-8"))
OUT_PATH = os.path.join(REPORT_DIR, "实验三 贝叶斯分类与BP算法.docx")

NUM = K.Numbering("3")


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
    K.cover_table(doc, date="2025.11.28")
    K.doc_title(doc, "实验三　贝叶斯分类与BP算法")

    # ================================================================= 一
    K.h1(doc, "一、贝叶斯分类器")
    K.h2(doc, "1.实验目标")
    K.body(doc, "掌握贝叶斯分类器的原理与应用方法。")

    K.h2(doc, "2、实验数据")
    K.body(doc, "过滤垃圾邮件数据(email datasets)：包含50条样本数据，每条数据为一个邮件，"
                "属于2个不同的类别。")

    K.h2(doc, "3、实验要求")
    for s in ["a.编写Python代码实验贝叶斯分类器。",
              "b.加载数据集后，进行基本的数据清洗和处理",
              "c.将数据集分为训练集和测试集，使用K折交叉验证来评估模型效果。",
              "d.训练贝叶斯分类器并对测试集进行预测。",
              "e.计算并报告模型的准确率、混淆矩阵、精确率、召回率和F1分数。",
              "f.绘制分类结果的混淆矩阵图。"]:
        K.body(doc, s)

    K.h2(doc, "4.实验步骤")
    K.body(doc, "本实验在 Python 3.12 环境下完成：pandas 负责数据整理，scikit-learn 提供"
                "朴素贝叶斯、文本向量化与神经网络实现。首先导入依赖库：")
    K.code_box(doc, code(1))

    # ---- a
    K.req_box(doc, "a.编写Python代码实验贝叶斯分类器。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(3))
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
    fig(doc, "theory/fig3_bayes_principle.jpg", "贝叶斯决策的要素与“朴素”条件独立假设")
    K.body(doc, "该假设在现实中显然不完全成立（词与词之间存在搭配关系），但它使需要估计的"
                "参数量由指数级降为线性级。本实验的邮件样本只有 50 条，正是这一简化让模型"
                "在小样本下依然可用。")
    K.body(doc, f"式 {e4} 存在一个致命缺陷：若某个词在某一类的训练样本中从未出现，其条件"
                "概率会被估成 0，连乘之后整封邮件的后验概率都变为 0。为此引入拉普拉斯平滑，"
                "给每个词的计数加一个正数 α、分母加上 αV（V 为词表大小）：")
    e5 = eq(doc, "f3_laplace")
    K.body(doc, f"式 {e5} 保证任何词的条件概率都不为零。下图用示意数据说明零概率问题及其"
                "修正效果。")
    fig(doc, "theory/fig3_laplace_smoothing.jpg", "零概率问题与拉普拉斯平滑")
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
                "两个类别各 25 封，样本数量完全均衡（图 3-3a）。读取时按目录来源把标签设为 "
                "0（正常邮件）与 1（垃圾邮件），并把文件内容与标签组装成 DataFrame。")
    K.body(doc, "清洗工作主要是统一大小写：英文文本中 “Free” 与 “free” 若被当作两个不同的"
                "特征，会稀释每个特征的统计量，在样本量很小的情况下影响尤为明显。转小写后"
                "词表规模减小、参数估计更稳定；此外，标点等符号在向量化阶段会由分词规则"
                "自然过滤，停用词则在 TF-IDF 中通过 stop_words=\"english\" 去除。")
    K.body(doc, "两类邮件的长度分布非常接近（图 3-3b，平均约 47 词与 42 词），说明邮件长度"
                "本身不构成判别依据，模型必须从用词差异中学习。")
    fig(doc, "theory/fig3_email_overview.jpg", "邮件数据集的类别分布与文本长度分布")
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
    K.label(doc, "【原理与推导】")
    K.body(doc, "文本无法直接输入模型，需要先向量化为数值特征。代码使用 TF-IDF 把每封邮件"
                "表示成词权向量，某个词在邮件中出现得越多、在整个语料中出现得越少，其权重"
                "越大：")
    e7 = eq(doc, "f3_tfidf")
    K.body(doc, f"式 {e7} 中 tf(t, d) 为词 t 在邮件 d 中的频次，df(t) 为包含该词的邮件数，"
                "N 为邮件总数。TF-IDF 权重非负，与多项式朴素贝叶斯的计数假设相容；像 "
                "“the”“is” 这类到处都出现的词因 df 很大而权重被压得很低。")
    K.body(doc, "随后把向量化与分类器封装成 Pipeline，再按 8 : 2 分层划分训练集与测试集。"
                "使用 Pipeline 的关键作用是：交叉验证的每一折中，TF-IDF 只在训练部分上"
                "拟合（fit），不会“看到”验证部分的词频统计，从而避免信息泄漏。")
    K.body(doc, "K 折交叉验证把数据集分成 K 份，轮流用其中一份作验证集、其余 K−1 份作训练集，"
                "最终以 K 个结果的均值作为性能估计：")
    e8 = eq(doc, "f3_cv")
    K.label(doc, "【运行结果】")
    K.body(doc, "5 折的准确率分别为 100%、80%、90%、90%、100%，平均 92.00%。折间波动高达 "
                "20 个百分点，原因是样本总量仅 50 条，每折只有 10 封邮件，判错一封就会带来 "
                "10 个百分点的变化。")
    K.out_box(doc, out_text(11))
    fig(doc, "theory/fig3_cv_scores.jpg", "5 折交叉验证的每折准确率与平均准确率")

    # ---- d
    K.req_box(doc, "d.训练贝叶斯分类器并对测试集进行预测。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(13))
    K.code_box(doc, code(14))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"训练阶段并不需要迭代优化，只是统计参数：按式 {e3} 估计先验、按式 {e5} "
                f"估计各词在两类下的条件概率；预测阶段则按式 {e6} 累加对数后验并取最大者。"
                "在对数域上可以看出，判决完全由各词的“对数似然比”共同决定，因此可以直接"
                "比较同一个词在两个类别下的权重差，把它作为该词判别能力的度量：")
    e9 = eq(doc, "f3_odds")
    K.body(doc, f"式 {e9} 取值越大说明该词越指向垃圾邮件，越小则越指向正常邮件。"
                "下图给出本实验训练结果中判别性最强的两组词。")
    fig(doc, "theory/fig3_top_words.jpg", "对数似然比最大（判为垃圾邮件）与最小（判为正常邮件）的词")
    K.label(doc, "【运行结果】")
    K.body(doc, "训练完成后输出所用估计器为 MultinomialNB()；模型对测试集的预测结果在下一项"
                "中统一评估。")
    K.out_box(doc, out_text(13))

    # ---- e
    K.req_box(doc, "e.计算并报告模型的准确率、混淆矩阵、精确率、召回率和F1分数。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(16))
    K.code_box(doc, code(17))
    K.code_box(doc, code(18))
    K.label(doc, "【原理与推导】")
    K.body(doc, "准确率与误差率互补，是最直观的指标：")
    e10 = eq(doc, "f3_error")
    K.body(doc, "但准确率无法区分两类错误，因此在文本分类中还需要混淆矩阵以及由它导出的"
                "查准率、查全率与 F1 分数：")
    e11 = eq(doc, "f3_prf")
    K.body(doc, f"式 {e11} 中，就垃圾邮件识别而言，查准率反映“被判为垃圾邮件的邮件中有多少"
                "真的是垃圾邮件”（是否误伤正常邮件），查全率反映“真正的垃圾邮件被拦截了"
                "多少”（是否漏判），F1 则是二者的调和平均。")
    K.body(doc, f"需要说明的是，本实验测试集只有 10 封邮件（正常邮件与垃圾邮件各 5 封），"
                f"按式 {e10} 计算，判错任何一封都会使准确率下降 10 个百分点。因此 100% 的"
                "测试准确率不宜单独解读，应结合 5 折交叉验证的平均准确率 92.00% 一起看，"
                "才能对模型的真实水平有客观判断。")
    K.label(doc, "【运行结果】")
    K.body(doc, "测试集准确率为 100%：")
    K.out_box(doc, out_text(16))
    K.body(doc, "混淆矩阵为对角阵，10 封邮件全部判对（正常邮件 5 封、垃圾邮件 5 封）：")
    K.out_box(doc, out_text(17))
    K.body(doc, "分类报告显示两类的查准率、查全率与 F1 分数均为 1.00，宏平均与加权平均也"
                "同为 1.00：")
    K.out_box(doc, out_text(18))

    # ---- f
    K.req_box(doc, "f.绘制分类结果的混淆矩阵图。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(20))
    K.label(doc, "【原理与推导】")
    K.body(doc, "混淆矩阵图把四个计数放进同一张热力图：对角线颜色越深说明判对的样本越多，"
                "非对角线则对应把正常邮件误判为垃圾邮件、或把垃圾邮件漏判这两类错误。"
                "在垃圾邮件过滤这类应用中，两类错误的代价并不相同——漏判只是多收到一封"
                "骚扰邮件，误判则可能造成重要邮件丢失，因此实际部署时往往需要结合代价"
                "调整判决阈值，而不能只看准确率。")
    K.label(doc, "【运行结果】")
    fig(doc, "concept/fig3_nb_confusion.jpg", "朴素贝叶斯在邮件测试集上的混淆矩阵")

    # ================================================================= 二
    doc.add_page_break()
    K.h1(doc, "二、BP算法")
    K.h2(doc, "1.实验目标")
    K.body(doc, "掌握BP算法的原理及其应用。")

    K.h2(doc, "2. 实验数据")
    K.body(doc, "考虑UCI数据库中关于红酒品质的数据集(Win Quality Data Set)。该数据集包含"
                "1599个红葡萄酒样本信息，记录了非挥发性酸含量、挥发性酸含量、柠檬酸含量等"
                "11个特征，响应变量为综合品质得分，取值0~10，此样本集仅包含3~8这6种评分。")

    K.h2(doc, "3. 实验要求")
    for s in ["a.读入数据；",
              "b.将红酒分为两个等级，其中评分3、4、5的为”bad”品质，评分为6、7、8的分为“good”品质。",
              "c. 将特征变量进行归一化处理。",
              "d. 设置一个随机种子，将数据集按7：3的比例划分为训练集和测试集。",
              "e.使用BP算法在训练数据集上构建合适的神经网络模型，隐藏层神经元个数设置为2.",
              "f.在测试集上计算模型的测试误差率。",
              "g.选取不同的隐藏层神经元个数构建神经网络模型，根据不同模型的测试错误率"
              "选出最优模型。"]:
        K.body(doc, s)

    K.h2(doc, "4.实验步骤")

    # ---- a
    K.req_box(doc, "a.读入数据；")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(23))
    K.label(doc, "【原理与推导】")
    K.body(doc, "红酒品质数据集共 1599 条样本，包含 11 个理化指标特征（非挥发性酸、挥发性酸、"
                "柠檬酸、残糖、氯化物、游离二氧化硫、总二氧化硫、密度、pH、硫酸盐、酒精）"
                "与 1 个品质得分 quality，数据完整、没有缺失值。quality 的实际取值为 3~8 分，"
                "其中 5 分（681 条）和 6 分（638 条）占绝大多数，3 分与 8 分分别只有 10 条和 "
                "18 条，分布明显集中。")
    K.label(doc, "【运行结果】")
    K.out_box(doc, out_text(23))
    fig(doc, "theory/fig3_quality_distribution.jpg", "红酒综合品质分布与二分类划分")

    # ---- b
    K.req_box(doc, "b.将红酒分为两个等级，其中评分3、4、5的为”bad”品质，"
                   "评分为6、7、8的分为“good”品质。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(25))
    K.label(doc, "【原理与推导】")
    K.body(doc, "把 6 档评分合并为二分类后，bad 类（3~5 分）共 744 条、占 46.5%，"
                "good 类（6~8 分）共 855 条、占 53.5%，两类比例接近，属于较均衡的二分类"
                "问题。因此后续可以直接用误差率衡量模型效果，同时也可以把“全部预测为 "
                "good”的 46.5% 误差率作为参照基准。")
    K.body(doc, "需要指出的是，把有序的品质评分强行切成两类会损失一部分信息：5 分（bad 类"
                "上限）与 6 分（good 类下限）在口感上差异很小，位于分界附近的样本本身就"
                "难以区分，这构成了本题误差率的下限。")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只新增 grade 与 label 两列，没有输出。")

    # ---- c
    K.req_box(doc, "c. 将特征变量进行归一化处理。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(27))
    K.code_box(doc, code(28))
    K.label(doc, "【原理与推导】")
    K.body(doc, "11 个特征的量纲差异极大：总二氧化硫的取值可达数百，氯离子含量只有 0.01 量级，"
                "密度在 0.99 附近，柠檬酸的取值甚至包含 0。神经网络使用梯度下降训练，"
                "若不做处理，参数更新会被量纲大的特征主导，收敛缓慢甚至发散。"
                "MinMaxScaler 把每个特征线性映射到 [0, 1]：")
    e12 = eq(doc, "f3_minmax")
    K.body(doc, f"式 {e12} 只依赖特征的极值，不改变样本之间的相对次序，因此不会影响特征与"
                "标签之间的关系。")
    fig(doc, "theory/fig3_scaling_effect.jpg", "归一化前后各特征取值的分布对比")
    K.body(doc, "这里有一个细节需要注意：cell 28 虽然对全体数据做了归一化，但该结果并未参与"
                "建模；真正用于训练与预测的是 cell 31 在划分之后、只用训练集 fit 的 scaler。"
                "这样处理是正确的，因为若先用全体数据（含测试集）统计极值，测试集的信息就"
                "会泄漏到训练过程中。")
    K.label(doc, "【运行结果】")
    K.body(doc, "两个单元格都不产生输出。")

    # ---- d
    K.req_box(doc, "d. 设置一个随机种子，将数据集按7：3的比例划分为训练集和测试集。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(30))
    K.code_box(doc, code(31))
    K.label(doc, "【原理与推导】")
    K.body(doc, "代码设置 random_state = 42，使每次运行得到完全相同的划分，保证结果可复现；"
                "同时启用 stratify 参数做分层采样，使训练集与测试集中 bad/good 的比例与整体"
                "保持一致。1599 条样本按 7 : 3 划分后，训练集 1119 条、测试集 480 条：")
    e13 = eq(doc, "f3_split")
    K.body(doc, f"如式 {e13} 所示，训练集用于拟合网络参数，测试集只在评估阶段使用，"
                "两者互不重叠。")
    K.label(doc, "【运行结果】")
    K.body(doc, "两个单元格都不产生输出；由样本总数可推知划分结果为训练集 1119 条、"
                "测试集 480 条。")

    # ---- e
    K.req_box(doc, "e.使用BP算法在训练数据集上构建合适的神经网络模型，隐藏层神经元个数设置为2.")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(33))
    K.label(doc, "【原理与推导】")
    K.body(doc, "神经网络的基本单元是神经元：它对输入特征做加权求和，再经激活函数得到输出：")
    e14 = eq(doc, "f3_neuron")
    K.body(doc, f"式 {e14} 中的激活函数 f 赋予网络非线性能力。本实验使用 ReLU：")
    e15 = eq(doc, "f3_relu")
    K.body(doc, f"式 {e15} 在正区间导数为常数 1，能有效缓解 Sigmoid 类函数在饱和区梯度趋近"
                "于 0 的问题；作为对比，Sigmoid 的表达式为：")
    e16 = eq(doc, "f3_sigmoid")
    fig(doc, "theory/fig3_neuron_activation.jpg", "神经元结构与常用激活函数")
    K.body(doc, "BP 算法包含两个过程。前向传播逐层计算各层输出，直到得到网络的预测值：")
    e17 = eq(doc, "f3_forward")
    K.body(doc, f"随后用交叉熵衡量预测与真实标签之间的偏差（式 {e17} 中 a 即各层输出）：")
    e18 = eq(doc, "f3_loss")
    K.body(doc, f"如式 {e18} 所示，损失由两部分组成：第一项是损失对输出的偏导，第二项是"
                "激活函数的导数。反向传播则利用链式法则把误差从输出层逐层向前传递，"
                "先求输出层的误差项：")
    e19 = eq(doc, "f3_delta_out")
    K.body(doc, f"再由后一层的误差项递推得到各隐藏层的误差项：")
    e20 = eq(doc, "f3_delta_hidden")
    K.body(doc, f"式 {e19}、式 {e20} 是 BP 算法的核心。得到各层误差项后，参数的梯度可以写成"
                "误差项与本层输入的乘积，并按梯度下降更新：")
    e21 = eq(doc, "f3_grad")
    e22 = eq(doc, "f3_update")
    fig(doc, "theory/fig3_bp_flow.jpg", "BP 网络的前向传播与误差反向传播")
    K.body(doc, f"式 {e21}、式 {e22} 即一次完整的参数更新过程。代码中 "
                "MLPClassifier(hidden_layer_sizes=(2,), activation=\"relu\", solver=\"adam\", "
                "max_iter=1000, random_state=42) 建立的是一个 11-2-2 的三层网络：输入层 11 个"
                "特征、隐藏层 2 个神经元、输出层 2 个类别，使用 Adam 优化器最多迭代 1000 轮。")
    K.label(doc, "【运行结果】")
    K.out_box(doc, out_text(33))

    # ---- f
    K.req_box(doc, "f.在测试集上计算模型的测试误差率。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(35))
    K.code_box(doc, code(36))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"测试误差率与准确率互补，即式 {e10}。误差率越低说明模型的泛化能力越好。")
    K.body(doc, "下图给出隐藏层为 2、1、10 个神经元时的训练损失下降曲线。可以看出损失都在"
                "前几十轮迅速下降，随后趋于平缓，说明 Adam 优化器收敛较快；但损失下降得"
                "低并不等于测试效果好，网络仍可能把训练集“背下来”而在测试集上失效。")
    fig(doc, "theory/fig3_loss_curves.jpg", "不同隐藏层规模的训练损失下降过程")
    K.label(doc, "【运行结果】")
    K.body(doc, "隐藏层 2 个神经元的模型测试误差率为 54.17%，即准确率仅 45.83%，甚至低于"
                "“全部预测为 good”的 46.5% 误差率：")
    K.out_box(doc, out_text(36))
    K.body(doc, "从混淆矩阵可以看清退化的原因：该网络把所有测试样本都判为 bad，"
                "257 个 good 类样本无一被识别出来，说明隐藏层的 2 个 ReLU 单元在该次训练中"
                "几乎全部失活，网络退化成了常数分类器。")
    fig(doc, "concept/fig3_bp_confusion.jpg", "隐藏层 2 个神经元的 BP 网络在测试集上的混淆矩阵")

    # ---- g
    K.req_box(doc, "g.选取不同的隐藏层神经元个数构建神经网络模型，根据不同模型的测试错误率"
                   "选出最优模型。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(38))
    K.code_box(doc, code(39))
    K.label(doc, "【原理与推导】")
    K.body(doc, "隐藏层神经元个数决定了网络的表示能力：单元太少会欠拟合，太多则容易过拟合，"
                "同时非凸优化也更容易落到较差的局部最优。为此对 1、2、3、5、10、15 六种规模"
                "分别训练并比较测试误差率。")
    K.body(doc, "从结果看，除 2 个神经元外，其余规模的测试误差率都稳定在 26.0%~26.7% 之间，"
                "彼此差异不到 1 个百分点。这说明当前限制模型效果的主要因素并不是网络容量，"
                "而是 11 个理化指标与品质等级之间本身就存在的重叠——由 5 分与 6 分构成的"
                "分界线附近样本天然难以区分。")
    K.body(doc, "2 个神经元模型的误差率高达 54.17%，明显偏离上述水平。这说明在非凸的损失"
                "曲面上，不同隐藏层规模会收敛到不同的局部最优：该规模下两个 ReLU 单元同时"
                "失活，网络退化为常数分类器，与本项要求 f 中观察到的现象一致。因此报告中的"
                "“最优模型”是按测试误差率选出的 1 个神经元模型，但更稳妥的结论是："
                "在本数据上 3 个以上神经元都能取得相近效果，2 个神经元是需要避免的配置。")
    K.body(doc, "此外还需注意，测试集只有 480 条样本，26.67% 与 26.04% 之间的差异对应不到 "
                "2 个样本，属于随机波动的范围，不宜据此断言某个规模更优。")
    fig(doc, "theory/fig3_hidden_sweep.jpg", "隐藏层神经元个数对测试错误率的影响")
    K.label(doc, "【运行结果】")
    K.body(doc, "六种规模的实验结果如下：")
    K.out_box(doc, out_text(38))
    K.body(doc, "按测试误差率最小选出的最优模型为隐藏层 1 个神经元（误差率 26.04%，"
                "准确率 73.96%）：")
    K.out_box(doc, out_text(39))

    doc.save(OUT_PATH)
    print("已生成：", OUT_PATH)
    print(f"插图 {NUM.fig} 张，公式 {NUM.eq} 个")


if __name__ == "__main__":
    build()
