# -*- coding: utf-8 -*-
"""实验四实验报告生成脚本（集成模型：AdaBoost 与决策树非均衡分类）。

风格与实验一/二/三一致：只写"4.实验步骤"，每项要求按
【实验要求】框 → 【代码实现】代码框 → 【原理与推导】 → 【运行结果】组织；
不写 a/b/c/d 小标题；图注不带来源/主题后缀；代码直接从 exp4.ipynb 读取。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docx_kit as K

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
ASSETS = os.path.join(REPORT_DIR, "素材", "exp4")
NB = json.load(open(os.path.join(PROJ_DIR, "exp4.ipynb"), encoding="utf-8"))
OUT_PATH = os.path.join(REPORT_DIR, "实验四 集成模型.docx")

NUM = K.Numbering("4")


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
    K.cover_table(doc, date="2025.12.05")
    K.doc_title(doc, "实验四　集成模型")

    # ================================================================= 一
    K.h1(doc, "一、AdaBoost集成学习")
    K.h2(doc, "1.实验目标")
    K.body(doc, "掌握集成学习的原理与应用方法")

    K.h2(doc, "2. 实验数据")
    K.body(doc, "使用一个二分类任务的信用卡欺诈测试数据集(creditcard.csv)，该数据集包含用户的"
                "交易信息，能够识别欺诈行为。Credit_card数据集共有17个特征，credit_card_label"
                "数据集为样例标签，表示交易是否欺诈，0为正常交易，1为欺诈交易。")

    K.h2(doc, "3. 实验要求")
    for s in ["a.编写Python代码实现AdaBoost模型，选择一个基本分类器作为弱学习器。",
              "b.读取数据集，处理缺失值（如有），并将数据集分为训练集和测试集。",
              "c.在训练集上训练AdaBoost模型。在测试集上进行预测，并计算准确率。",
              "d.打印混淆矩阵，打印分类报告，显示每个类别的精确率、召回率和F1分数。",
              "e.绘制混淆均值的热力图。"]:
        K.body(doc, s)

    K.h2(doc, "4.实验步骤")
    K.body(doc, "本实验在 Python 3.12 环境下完成：pandas 负责数据读取与合并，scikit-learn 提供"
                "数据预处理、集成学习与决策树实现，seaborn 用于绘制热力图。首先导入依赖库：")
    K.code_box(doc, code(1))

    # ---- a
    K.req_box(doc, "a.编写Python代码实现AdaBoost模型，选择一个基本分类器作为弱学习器。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(3))
    K.label(doc, "【原理与推导】")
    K.body(doc, "集成学习的思路是“三个臭皮匠顶个诸葛亮”：把若干个性能一般的弱学习器按一定"
                "方式组合起来，得到一个更强的模型。AdaBoost 属于其中最具代表性的提升"
                "（boosting）方法，它串行地训练一系列弱学习器，最终的判决由它们加权投票得到：")
    e1 = eq(doc, "f4_ensemble")
    K.body(doc, "本实验选用的弱学习器是决策树桩，即深度为 1 的决策树，它只能做一次"
                "“一刀切”的判断，性能很弱但计算极快：")
    e2 = eq(doc, "f4_stump")
    K.body(doc, f"AdaBoost 的训练是一个逐轮迭代的过程。第 t 轮先计算当前弱学习器在加权样本"
                "上的错误率：")
    e3 = eq(doc, "f4_eps")
    K.body(doc, f"再由错误率决定该学习器在最终投票中的话语权：错误率越小，权重越大；当"
                f"错误率恰为 0.5（与随机猜测无异）时权重为 0。")
    e4 = eq(doc, "f4_alpha")
    K.body(doc, f"随后提高被这一轮判错样本的权重，使下一轮的弱学习器更关注难分样本：")
    e5 = eq(doc, "f4_wupdate")
    K.body(doc, "其中分母为归一化因子，保证更新后样本权重之和仍为 1：")
    e6 = eq(doc, "f4_Z")
    fig(doc, "theory/fig4_adaboost_principle.jpg",
        "弱学习器的加权组合与权重随错误率的变化")
    K.body(doc, f"式 {e1}~式 {e6} 构成本实验的模型基础。代码中 "
                "DecisionTreeClassifier(max_depth=1, random_state=42) 建立的正是一个决策树桩；"
                "固定随机种子是为了保证结果可复现。")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建弱学习器对象，尚未训练，因此没有输出。")

    # ---- b
    K.req_box(doc, "b.读取数据集，处理缺失值（如有），并将数据集分为训练集和测试集。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(5))
    K.code_box(doc, code(6))
    K.code_box(doc, code(7))
    K.body(doc, "特征中包含数值列与类别列，二者需要分别处理，因此先把缺失值填充与编码规则"
                "写成一个预处理组件：")
    K.code_box(doc, code(8))
    K.code_box(doc, code(9))
    K.label(doc, "【原理与推导】")
    K.body(doc, "交易特征表 Credit_card.csv 共 1548 行、17 个特征，标签表 Credit_card_label.csv "
                "给出对应的 label，两表通过唯一键 Ind_ID 做内连接合并。原始特征表中共有 540 个"
                "缺失值，需要先处理：数值特征用中位数填充（对极端值不敏感），类别特征用众数"
                "填充：")
    e7 = eq(doc, "f4_impute")
    K.body(doc, f"式 {e7} 之后，8 个类别型特征（性别、车辆拥有、房产、收入类型、学历、"
                "婚姻状况、住房类型、职业类型）无法直接参与计算，需做 One-Hot 编码：")
    e8 = eq(doc, "f4_onehot")
    K.body(doc, f"如式 {e8} 所示，含 V 个取值的类别特征被展开为 V 个 0/1 稀疏列。把填充与"
                "编码封装进 ColumnTransformer，再与分类器一起放进 Pipeline，可以保证"
                "交叉验证或划分时预处理参数只在训练部分上估计，避免信息泄漏。整体流程如下：")
    fig(doc, "theory/fig4_pipeline.jpg", "数据预处理与建模流水线")
    K.body(doc, "标签分布存在明显的不平衡：正常交易 1373 笔（88.70%）、欺诈交易 175 笔"
                "（11.30%），正负样本比约为 1 : 7.8：")
    e9 = eq(doc, "f4_ratio")
    K.body(doc, f"式 {e9} 中的不平衡比越高，模型越容易倒向多数类。因此在按 7 : 3 分层划分"
                "训练集与测试集之后，测试集为 465 笔、训练集为 1083 笔。同时需要记住一个"
                "参照基准——若模型不加学习、一律预测为“正常交易”，测试集准确率仍可达到：")
    e10 = eq(doc, "f4_baseline")
    fig(doc, "theory/fig4_imbalance_baseline.jpg", "类别不平衡与常数基线准确率")
    K.label(doc, "【运行结果】")
    K.body(doc, "以上单元格只完成数据读取、合并、预处理定义与划分，均不产生输出；"
                "由样本总数与划分比例可知训练集 1083 笔、测试集 465 笔。")

    # ---- c
    K.req_box(doc, "c.在训练集上训练AdaBoost模型。在测试集上进行预测，并计算准确率。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(11))
    K.code_box(doc, code(12))
    K.code_box(doc, code(13))
    K.code_box(doc, code(14))
    K.label(doc, "【原理与推导】")
    K.body(doc, "模型由预处理与 AdaBoost 两部分串联而成：AdaBoostClassifier(estimator="
                "base_classifier, n_estimators=100, learning_rate=1.0) 表示串行训练 100 个"
                "决策树桩，学习率 1.0 即直接使用式 " + e4.strip("()") + " 计算出的权重。"
                "评价使用的准确率为判对样本占样本总数的比例：")
    e11 = eq(doc, "f4_acc")
    K.body(doc, f"结合式 {e10} 的基线可以看出，在不平衡数据上“准确率”本身就是一个容易"
                "产生误导的指标：只要模型倾向于预测多数类，就能取得很高的准确率。")
    K.label(doc, "【运行结果】")
    K.body(doc, "模型在 465 笔测试交易上的准确率为 88.17%：")
    K.out_box(doc, out_text(14))
    K.body(doc, "这一数值低于“全部预测为正常交易”的 88.60%，说明单看准确率，集成模型"
                "甚至不如什么都不学。原因将在下一项的分类报告中暴露。")

    # ---- d
    K.req_box(doc, "d.打印混淆矩阵，打印分类报告，显示每个类别的精确率、召回率和F1分数。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(16))
    K.code_box(doc, code(17))
    K.label(doc, "【原理与推导】")
    K.body(doc, "混淆矩阵把预测结果与真实标签交叉统计，由它可以直接导出查准率、查全率与 "
                "F1 分数：")
    e12 = eq(doc, "f4_prf")
    K.body(doc, f"式 {e12} 中，查准率回答“被判为欺诈的交易中有多少是真欺诈”（是否误伤"
                "正常用户），查全率回答“真正的欺诈交易被拦截了多少”（是否漏放），"
                "F1 是二者的调和平均。多分类或需要综合各类表现时还可使用宏平均与加权平均：")
    e13 = eq(doc, "f4_macro")
    K.body(doc, f"式 {e13} 中，宏平均对每个类别同等看待，因此少数类的表现会被放大；"
                "加权平均则按各类样本数加权，容易被多数类主导。在不平衡数据上，"
                "宏平均 F1 往往比准确率更能反映模型的真实水平。")
    fig(doc, "concept/fig4_metrics_bars.jpg", "AdaBoost 各类别的查准率、查全率与 F1 分数")
    K.label(doc, "【运行结果】")
    K.body(doc, "混淆矩阵显示：412 笔正常交易中 410 笔判对、2 笔误判为欺诈；"
                "而 53 笔欺诈交易全部被判为正常，无一被识别：")
    K.out_box(doc, out_text(16))
    K.body(doc, "分类报告进一步印证了这一点：Fraud 类的查准率、查全率与 F1 分数全部为 0，"
                "宏平均 F1 仅 0.47，而按样本数加权的平均 F1 却高达 0.83——两者相差悬殊，"
                "正是类别不平衡下“准确率虚高”的典型表现：")
    K.out_box(doc, out_text(17))
    K.body(doc, "这一现象的原因在于：AdaBoost 的 100 个决策树桩以最小化整体加权错误率为"
                "目标，而欺诈样本只占 11.30%，模型只要把所有样本判为正常，就能把总错误率"
                "压到很低；同时模型也未设置类别权重（class_weight）来抵消这种倾向。"
                "在业务上，漏放一笔欺诈的损失通常远大于误伤一笔正常交易，因此这类任务"
                "必须结合查全率与业务代价来选择模型，而不能以准确率作为优化目标。")

    # ---- e
    K.req_box(doc, "e.绘制混淆均值的热力图。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(19))
    K.label(doc, "【原理与推导】")
    K.body(doc, "热力图把混淆矩阵的四个计数用颜色深浅表示：对角线越深说明判对的样本越多，"
                "非对角线则对应两类错误。对本实验而言，最值得注意的是右下角那一格——"
                "它本应记录被成功识别的欺诈交易，实际数值为 0，说明模型在该类上完全失效。")
    K.label(doc, "【运行结果】")
    K.add_fig(doc, J("notebook_out", "cell19.jpg"), "AdaBoost 分类结果的混淆矩阵热力图", NUM)

    # ================================================================= 二
    doc.add_page_break()
    K.h1(doc, "二、决策树(非均衡分类问题)")
    K.h2(doc, "1.实验目标")
    K.body(doc, "a.理解决策树的基本原理及其应用。")
    K.body(doc, "b.掌握决策树模型的构建和评估方法。")

    K.h2(doc, "2. 实验数据")
    K.body(doc, "本实验采用图书购买数据集(Florence.cvs)，其中包含多个特征，"
                "用于预测顾客是否会购买《Florence》。")

    K.h2(doc, "3. 实验要求")
    for s in ["a.编写Python代码实现决策树模型。(考虑平衡因子)",
              "b.将数据集分为训练集和测试集，构建决策树模型并进行训练。",
              "c.使用测试集评估模型的性能，包括准确率、混淆矩阵、每个类别的精确率、"
              "召回率和F1分数等评估指标。",
              "d.随机选取3个特征，重新训练并可视化决策树。"]:
        K.body(doc, s)

    K.h2(doc, "4.实验步骤")
    K.body(doc, "先读入数据并查看结构：")
    K.code_box(doc, code(21))

    # ---- a
    K.req_box(doc, "a.编写Python代码实现决策树模型。(考虑平衡因子)")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(23))
    K.label(doc, "【原理与推导】")
    K.body(doc, "决策树通过一系列“如果……就……”的判断把特征空间递归地切成若干矩形区域，"
                "每个区域对应树的一个叶节点，叶节点的类别即为该区域的预测结果：")
    e14 = eq(doc, "f4_leaf")
    K.body(doc, "每次划分都要选择“最能降低不确定性”的特征与切分点。常用的不确定性度量有两种，"
                "熵与基尼指数：")
    e15 = eq(doc, "f4_entropy")
    e16 = eq(doc, "f4_gini")
    K.body(doc, f"式 {e15} 与式 {e16} 都在两类各占一半时取最大值（分别为 1 与 0.5）。"
                "本实验使用信息增益作为划分准则，即划分前后熵的下降量：")
    e17 = eq(doc, "f4_infogain")
    fig(doc, "theory/fig4_tree_principle.jpg", "决策树的递归划分与节点不纯度度量")
    K.body(doc, "若不加限制，决策树会一直划分到每个叶节点都纯，从而把训练集完全记住。"
                "代码中通过 max_depth=5、min_samples_split=10、min_samples_leaf=5 限制树的"
                "复杂度，这三者共同起到预剪枝的作用：")
    e18 = eq(doc, "f4_prune")
    K.body(doc, f"式 {e18} 中，max_depth 限制树的最大深度，min_samples_split 与 "
                "min_samples_leaf 分别限制内部节点继续划分所需的最少样本数和叶节点的最少"
                "样本数，二者可避免模型被个别样本牵着走。")
    K.body(doc, "本实验的关键参数是 class_weight=\"balanced\"，即“平衡因子”。它会按类别"
                "样本数的倒数自动给各类赋权：")
    e19 = eq(doc, "f4_classweight")
    K.body(doc, f"式 {e19} 中 N 为总样本数、K 为类别数、N_k 为第 k 类的样本数。样本越少的"
                "类别，其权重越大，从而使模型在划分时不再由多数类主导。")
    K.label(doc, "【运行结果】")
    K.body(doc, "该单元格只创建决策树对象，尚未训练，因此没有输出。")

    # ---- b
    K.req_box(doc, "b.将数据集分为训练集和测试集，构建决策树模型并进行训练。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(25))
    K.code_box(doc, code(26))
    K.label(doc, "【原理与推导】")
    K.body(doc, "Florence 数据集共 4000 条样本，包含 Seq#（序号）、Gender（性别）与 M、R、F "
                "三个消费行为特征，标签 Florence 表示是否购买，取值 0/1。其中未购买 3662 条"
                "（91.55%）、购买 338 条（8.45%），正负样本比约 1 : 10.8，属于典型的"
                "非均衡分类问题；按 8 : 2 分层划分后训练集 3200 条、测试集 800 条。")
    fig(doc, "theory/fig4_florence_overview.jpg", "Florence 数据集的类别分布与 RFM 特征分布")
    K.body(doc, "为说明“平衡因子”的实际作用，下面对同一份数据分别训练不使用类别权重与使用 "
                "balanced 权重的两棵决策树，其余参数完全一致：")
    fig(doc, "concept/fig4_balanced_compare.jpg", "是否启用平衡因子的效果对比")
    K.label(doc, "【运行结果】")
    K.body(doc, "模型拟合完成，输出所用估计器的参数（其中 class_weight='balanced' 即平衡因子）：")
    K.out_box(doc, out_text(26))
    K.body(doc, "对比实验很能说明问题：不加平衡因子时，模型准确率虽然高达 91.50%，但预测为"
                "“会购买”的样本数为 0，购买类的查准率、查全率与 F1 全为 0，等价于一个恒判"
                "“不购买”的常数分类器；启用 balanced 后准确率下降到 55.62%，但购买类的"
                "查全率提升到 0.56，即能识别出约一半的潜在购买者。在营销场景中后者才是有"
                "业务价值的模型，这也说明不平衡数据上必须结合业务目标选择模型，而不能只看"
                "准确率。")

    # ---- c
    K.req_box(doc, "c.使用测试集评估模型的性能，包括准确率、混淆矩阵、每个类别的精确率、"
                   "召回率和F1分数等评估指标。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(28))
    K.code_box(doc, code(29))
    K.code_box(doc, code(30))
    K.code_box(doc, code(31))
    K.label(doc, "【原理与推导】")
    K.body(doc, f"评价指标与第一部分的式 {e11}（准确率）、式 {e12}（查准率/查全率/F1）"
                f"完全一致。需要强调的是，在本数据上准确率的参照基准为 91.55%"
                f"（全部预测为“不购买”），因此 55.63% 的准确率看似很差，但它换来了少数类"
                " 0.56 的查全率，二者的取舍应由业务代价决定。")
    K.label(doc, "【运行结果】")
    K.body(doc, "模型在 800 条测试样本上的准确率为 55.625%：")
    K.out_box(doc, out_text(29))
    K.body(doc, "混淆矩阵显示：732 条未购买样本中 407 条判对、325 条被误判为会购买；"
                "68 条购买样本中 38 条被成功识别、30 条被漏判：")
    K.out_box(doc, out_text(30))
    K.body(doc, "分类报告显示：未购买类的查准率 0.93、查全率 0.56、F1 0.70，购买类的"
                "查准率 0.10、查全率 0.56、F1 0.18。购买类查准率偏低是因为模型把大量未购买"
                "用户也预测为会购买（325 例误报），这是启用平衡因子后的必然代价；"
                "宏平均 F1 为 0.44，加权平均 F1 为 0.65：")
    K.out_box(doc, out_text(31))

    # ---- d
    K.req_box(doc, "d.随机选取3个特征，重新训练并可视化决策树。")
    K.label(doc, "【代码实现】")
    K.code_box(doc, code(34))
    K.code_box(doc, code(35))
    K.code_box(doc, code(36))
    K.code_box(doc, code(37))
    K.code_box(doc, code(38))
    K.label(doc, "【原理与推导】")
    K.body(doc, "随机选取特征子集重新训练，可以观察模型对特征数量的敏感程度：若少量特征"
                "就能达到与全特征相近的效果，说明其余特征信息冗余；若性能明显下降，"
                "则说明被舍弃的特征包含了关键信息。代码先固定 random.seed(42) 再调用 "
                "random.sample 抽 3 个特征，保证抽取结果可复现。")
    K.body(doc, "为了解哪些特征真正起作用，下图给出全特征模型中各特征的重要性"
                "（按不纯度下降量加权统计）：")
    fig(doc, "theory/fig4_importance.jpg", "全特征决策树的特征重要性")
    K.body(doc, "图中 M（消费金额）的重要性高达 0.560，远超 F（频次）0.213、R（最近购买）"
                "0.171 与 Gender 0.056，说明“买了多少钱”是判断是否购买《Florence》的"
                "主导因素，这也与直觉一致。")
    K.body(doc, "决策树的一大优势是可解释性强，可以直接把树画出来观察判决逻辑。"
                "下面两图分别给出全特征模型与随机 3 特征模型的前三层结构"
                "（节点中给出划分条件与样本数，叶节点给出预测类别）：")
    fig(doc, "theory/fig4_tree_viz.jpg", "全特征决策树的前三层结构")
    fig(doc, "theory/fig4_tree_3feat.jpg", "随机 3 个特征决策树的前三层结构")
    K.label(doc, "【运行结果】")
    K.body(doc, "随机抽到的 3 个特征为 Gender、F、M：")
    K.out_box(doc, out_text(34))
    K.body(doc, "用这 3 个特征重新训练后，模型在测试集上的表现如下：准确率 0.5425，"
                "混淆矩阵显示 399 条未购买判对、33 条购买被识别；购买类的查准率 0.10、"
                "查全率 0.51、F1 0.16：")
    K.out_box(doc, out_text(38))
    K.body(doc, "与全特征模型的 55.63% 准确率、0.56 查全率相比，只用 3 个特征的模型性能"
                "略有下降但幅度很小（准确率下降约 1.4 个百分点，购买类 F1 由 0.18 降到 0.16），"
                "说明 M、F 这两个被抽中的特征已经承载了大部分判别信息；被舍弃的 R 与 "
                "Seq# 对结果影响有限。这也与前图中特征重要性的排序相互印证。")

    doc.save(OUT_PATH)
    print("已生成：", OUT_PATH)
    print(f"插图 {NUM.fig} 张，公式 {NUM.eq} 个")


if __name__ == "__main__":
    build()
