# -*- coding: utf-8 -*-
"""
实验一实验报告生成脚本（方案 B：生成新的 .docx，不改动原始 .doc）

排版约定：
  * 代码一律放在"代码框"（单元格表格 + 灰底 + 细边框）内，并做 Python 语法着色
  * 公式用 mathtext 渲染为高清图片，按 Word 居中制表位排版并自动编号 (1-n)
  * 原理插图全部按教材理论自绘（v2 起不再使用课件截图），统一编号 图 1-n
  * 只撰写"4、实验步骤"部分的内容，其余章节文字严格照抄原报告题目文件
"""
import io
import json
import keyword
import os
import builtins

import tokenize

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
ASSETS = os.path.join(REPORT_DIR, "素材")
NB_PATH = os.path.join(PROJ_DIR, "exp1.ipynb")
OUT_PATH = os.path.join(REPORT_DIR, "实验一 逻辑斯蒂回归与线性判别分析.docx")

CN_FONT, CN_HEI, MONO = "宋体", "黑体", "Consolas"
TEXT_W_CM = 16.0                      # 正文可用宽度（A4 减去左右 2.5cm 页边距）
EQ_SCALE = 0.62                       # 公式图片缩放比（原图按 20pt @300dpi 渲染）
FIG_MAX_IN = 5.9                      # 插图最大宽度（英寸）

nb = json.load(open(NB_PATH, encoding="utf-8"))


# ---------------------------------------------------------------------------
# 基础排版工具
# ---------------------------------------------------------------------------
def set_font(run, size=10.5, bold=False, cn=CN_FONT, ascii_font="Times New Roman",
             color=None, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = ascii_font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)
    rfonts.set(qn("w:eastAsia"), cn)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def para(doc, text="", size=10.5, bold=False, cn=CN_FONT, ascii_font="Times New Roman",
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent_chars=0, before=0, after=4,
         line=1.25, color=None, style=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if indent_chars:
        pf.first_line_indent = Pt(size * indent_chars)
    if text:
        run = p.add_run(text)
        set_font(run, size=size, bold=bold, cn=cn, ascii_font=ascii_font, color=color)
    return p


def h1(doc, text):
    return para(doc, text, size=14, bold=True, cn=CN_HEI, ascii_font=CN_HEI,
                align=WD_ALIGN_PARAGRAPH.LEFT, before=14, after=6, line=1.2)


def h2(doc, text):
    return para(doc, text, size=12, bold=True, cn=CN_HEI, ascii_font=CN_HEI,
                align=WD_ALIGN_PARAGRAPH.LEFT, before=10, after=5, line=1.2)


def h3(doc, text):
    return para(doc, text, size=10.5, bold=True, cn=CN_HEI, ascii_font=CN_HEI,
                align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=4, line=1.2)


def body(doc, text):
    return para(doc, text, indent_chars=2)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def cell_margins(table, top=60, start=100, bottom=60, end=100):
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for tag, val in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        el = OxmlElement("w:" + tag)
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr.append(mar)


def one_cell_box(doc, fill, border="BFBFBF"):
    """返回一个 1x1 单元格表格，作为代码框 / 要求框 / 结果框的容器。"""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    cell = t.cell(0, 0)
    cell.width = Cm(TEXT_W_CM)
    shade(cell, fill)
    cell_margins(t)
    # 边框颜色
    tblpr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:color"), border)
        borders.append(el)
    tblpr.append(borders)
    cell.paragraphs[0].text = ""
    return cell


# ---------------------------------------------------------------------------
# Python 语法着色
# ---------------------------------------------------------------------------
def py_segments(code):
    """把源码切成 (文本, 颜色) 片段，颜色取自浅色主题的语法高亮方案。

    实现要点：只对"有实体内容的 token"着色，其余字符（空白、换行、缩进）
    直接按原始文本原样输出，因此拼接结果与源码逐字符一致。
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception:
        return [(code, "1F1F1F")]

    line_start, off = [], 0
    for ln in code.split("\n"):
        line_start.append(off)
        off += len(ln) + 1

    def to_off(pos):
        r, c = pos
        if 1 <= r <= len(line_start):
            return line_start[r - 1] + c
        return len(code)

    skip = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE,
            tokenize.NL, tokenize.INDENT, tokenize.DEDENT}
    segs, cursor, pending_def = [], 0, False
    for t in toks:
        if t.type in skip or not t.string:
            continue
        color = "1F1F1F"
        if t.type == tokenize.COMMENT:
            color = "008000"
        elif t.type == tokenize.STRING:
            color = "A31515"
        elif t.type == tokenize.NUMBER:
            color = "098658"
        elif t.type == tokenize.NAME:
            if keyword.iskeyword(t.string):
                color = "0000FF"
                pending_def = t.string in ("def", "class")
            elif pending_def:
                color = "267F99"
                pending_def = False
            elif t.string in dir(builtins):
                color = "795E26"
        s, e = to_off(t.start), to_off(t.end)
        if s > cursor:
            segs.append((code[cursor:s], "1F1F1F"))
        segs.append((code[s:e], color))
        cursor = e
    if cursor < len(code):
        segs.append((code[cursor:], "1F1F1F"))
    return segs


def code_box(doc, code):
    """把一段代码放进带灰底与边框的代码框，逐 token 着色。"""
    cell = one_cell_box(doc, "F7F7F7")
    cell.paragraphs[0]._element.getparent().remove(cell.paragraphs[0]._element)
    lines = [[]]
    for text, color in py_segments(code.rstrip("\n")):
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k:
                lines.append([])
            if part:
                lines[-1].append((part, color))
    for ln in lines:
        p = cell.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = Pt(12)
        for text, color in ln:
            run = p.add_run(text if text.strip() else text)
            set_font(run, size=9, cn=CN_FONT, ascii_font=MONO, color=color)
    return cell


def label(doc, text, color="1F4E79"):
    return para(doc, text, size=10.5, bold=True, cn=CN_HEI, ascii_font=CN_HEI,
                align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=3, color=color)


def req_box(doc, text):
    """【实验要求】原句引用框。"""
    cell = one_cell_box(doc, "EAF1F8", border="9DC3E6")
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.2
    r1 = p.add_run("实验要求　")
    set_font(r1, size=10.5, bold=True, cn=CN_HEI, ascii_font=CN_HEI, color="1F4E79")
    r2 = p.add_run(text)
    set_font(r2, size=10.5)
    return cell


def out_box(doc, text):
    """运行结果（控制台输出）框。"""
    cell = one_cell_box(doc, "FAFAFA", border="C9C9C9")
    cell.paragraphs[0]._element.getparent().remove(cell.paragraphs[0]._element)
    for ln in text.rstrip("\n").split("\n"):
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(12)
        run = p.add_run(ln if ln else " ")
        set_font(run, size=9, cn=CN_FONT, ascii_font=MONO, color="1F1F1F")
    return cell


# ---------------------------------------------------------------------------
# 公式与插图（自动编号）
# ---------------------------------------------------------------------------
class Numbering:
    def __init__(self):
        self.fig = 0
        self.eq = 0

    def next_fig(self):
        self.fig += 1
        return f"图 1-{self.fig}"

    def next_eq(self):
        self.eq += 1
        return f"(1-{self.eq})"


NUM = Numbering()


def add_eq(doc, key):
    """插入公式：普通段落 + 居中/右对齐制表位，实现“公式居中、编号右对齐”。

    说明：公式若放进表格单元格，LibreOffice/WPS 会按列宽裁切图片，
    因此这里改用制表位排版（已在 LibreOffice 渲染下验证尺寸准确）。
    """
    path = os.path.join(ASSETS, "formula", key + ".png")
    w, h = Image.open(path).size
    label_txt = NUM.next_eq()
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(5)
    pf.space_after = Pt(5)
    pf.line_spacing = 1.0
    pf.tab_stops.add_tab_stop(Cm(TEXT_W_CM / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT)
    scale = min(EQ_SCALE, 11.6 * 300 / (w * 2.54))      # 自动适配版心，避免公式超宽
    run = p.add_run()
    run.add_tab()
    run.add_picture(path, width=Cm(w / 300 * 2.54 * scale))
    run.add_tab()
    set_font(p.add_run(label_txt), size=10.5)
    return label_txt


def add_fig(doc, path, caption, max_in=5.7):
    """插入居中插图与图注，返回图号字符串。"""
    w, h = Image.open(path).size
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(path, width=Cm(max_in * 2.54))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(0)
    cap.paragraph_format.space_after = Pt(8)
    cap.paragraph_format.line_spacing = 1.0
    label_txt = NUM.next_fig()
    r = cap.add_run(f"{label_txt}　{caption}")
    set_font(r, size=9, color="404040")
    return label_txt


def nb_code(i):
    return "".join(nb["cells"][i]["source"])


def nb_out_text(i):
    parts = []
    for o in nb["cells"][i].get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o["text"]))
        elif o.get("output_type") == "execute_result":
            parts.append("".join(o["data"].get("text/plain", [])))
    return "".join(parts).strip("\n")


def nb_out_image(i, j=0):
    """返回 redraw_notebook.py 导出的白底插图（缺失时直接报错，避免误用深色旧图）。"""
    dst = os.path.join(ASSETS, "notebook_out", f"cell{i:02d}.jpg")
    if not os.path.exists(dst):
        raise FileNotFoundError(
            f"缺少白底插图 {dst}，请先运行 redraw_notebook.py 重新执行 notebook")
    return dst


def setup(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)          # A4，便于打印装订
    sec.page_height = Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.5)


def cover_table(doc):
    rows = [
        ("地　　点：", "B10 楼 301 房", "实验台号：", ""),
        ("实验日期与时间：", "2025.11.14", "评　　分：", ""),
        ("预习检查纪录：", "已预习", "实验教师：", "李成安"),
    ]
    t = doc.add_table(rows=3, cols=4)
    t.style = "Table Grid"
    widths = (3.4, 4.6, 3.4, 4.6)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.width = Cm(widths[c])
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(val)
            set_font(run, size=10.5, bold=(c % 2 == 0))
    return t


# ===========================================================================
# 正文构建
# ===========================================================================
def build():
    doc = Document()
    setup(doc)
    cover_table(doc)

    para(doc, "实验一　逻辑斯蒂回归　线性判别分析", size=16, bold=True, cn=CN_HEI,
         ascii_font=CN_HEI, align=WD_ALIGN_PARAGRAPH.CENTER, before=10, after=12)

    # ------------------------------------------------------------------ 一
    h1(doc, "一、逻辑斯蒂回归")

    h2(doc, "1. 实验目标")
    body(doc, "理解并实现 Logistic 回归算法；掌握如何使用逻辑回归解决实际问题。")

    h2(doc, "2、实验数据")
    body(doc, "Simmons 数据集（Simmons.csv）包含 50 名拥有 Simmons 信用卡的顾客和 50 名"
              "没有 Simmons 信用卡的其他顾客组成。目标变量是顾客是否使用了优惠券。")

    h2(doc, "3、实验要求")
    for s in ["a.编写Python代码实现Logistic 回归，并简单展示Sigmoid函数。",
              "b.读取数据集，并画出散点图。",
              "c.训练Logistic回归模型，并输出最终的分类准确率。",
              "d.绘制Logistic回归模型的决策边界。"]:
        body(doc, s)

    h2(doc, "4、实验步骤")

    # ---- 环境准备
    h3(doc, "（〇）实验环境与依赖库导入")
    body(doc, "本实验在 Python 3.12 环境下完成：使用 pandas 读取与整理数据，numpy 完成数值"
              "运算，matplotlib 负责绘图，scikit-learn 提供逻辑斯蒂回归、线性判别分析与"
              "各项评估指标的实现接口。首先导入全部依赖库：")
    code_box(doc, nb_code(1))

    # ---- a
    h3(doc, "a. 实现 Logistic 回归并展示 Sigmoid 函数")
    req_box(doc, "a.编写Python代码实现Logistic 回归，并简单展示Sigmoid函数。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(3))
    body(doc, "上述代码定义了 Sigmoid 函数并在区间 [-10, 10] 上取 200 个等距点求值，"
              "下面对该函数进行可视化：")
    code_box(doc, nb_code(4))

    label(doc, "【原理与推导】")
    body(doc, "Logistic 回归解决的是二分类问题。设样本为 d 维特征向量 x，其类别标记为 "
              "y ∈ {0, 1}，模型并不直接拟合类别，而是先构造特征的线性组合：")
    e1 = add_eq(doc, "f_linear")
    body(doc, f"如式 {e1} 所示，z 的取值覆盖整个实数轴，无法直接解释为概率。为此引入 "
              "Sigmoid 函数（亦称对数几率函数），把实数轴单调压缩到 (0, 1)：")
    e2 = add_eq(doc, "f_sigmoid")
    body(doc, f"式 {e2} 中 σ(z) 被解释为样本属于正类的后验概率 P(y = 1 | x)。由式 {e2} "
              "反解即可得到 Logistic 回归的名称来源——对数几率（log-odds）是输入的线性函数：")
    e3 = add_eq(doc, "f_logit")
    body(doc, f"式 {e3} 表明：若将正例判定的后验概率之比取对数，则它与特征呈线性关系，"
              "这正是 Logistic 回归被称为“对数几率回归”的原因。Sigmoid 函数还有一个"
              "在参数估计中极为关键的性质，即其导数可用自身表示：")
    e4 = add_eq(doc, "f_sigmoid_deriv")
    body(doc, f"式 {e4} 是后续梯度求解能够写成简洁形式的基础。下图给出 Sigmoid 函数与"
              "对数几率变换的直观图像。")
    add_fig(doc, os.path.join(ASSETS, "theory", "fig_sigmoid.jpg"),
            "Sigmoid 函数与对数几率变换（依据教材理论自绘）")
    body(doc, "从图中可见，当 z 远离 0 时 σ(z) 趋于饱和，导数接近 0，这正是深层网络中"
              "梯度消失的根源；因此在训练前对特征做标准化，使 z 的取值落在 Sigmoid 的"
              "线性区间附近，有利于参数收敛（本实验在第二部分对鸢尾花数据执行了标准化）。")

    label(doc, "【运行结果】")
    body(doc, "该单元格绘制出 Sigmoid 曲线（横轴 z，纵轴 Sigmoid(z)），并在 z = 0、"
              "Sigmoid(z) = 0.5 处标注参考线，与式 " + e2.strip("()") + " 的推导一致：")
    add_fig(doc, nb_out_image(4), "Sigmoid 函数图像（notebook 代码运行结果，白色主题）")

    # ---- b
    h3(doc, "b. 读取数据集并绘制散点图")
    req_box(doc, "b.读取数据集，并画出散点图。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(6))
    body(doc, "读取数据后打印数据结构信息，再以 Spending 为横轴、Card 为纵轴绘制散点图，"
              "并按是否使用优惠券（Coupon）区分颜色：")
    code_box(doc, nb_code(7))

    label(doc, "【原理与推导】")
    body(doc, "Simmons 数据集共 100 个样本，含 4 个字段：Customer（顾客编号）、"
              "Spending（消费额）、Card（信用卡持有情况）、Coupon（是否使用优惠券，"
              "为目标变量）。散点图的意义在于先在二维特征空间中观察两类样本的可分性："
              "若两类样本能够被一条直线大致分开，则线性模型（Logistic 回归、LDA）是合适"
              "的选择；若严重重叠，则分类准确率的上限将受到数据本身可分性的限制。")
    body(doc, "在划分训练集与测试集时，代码使用留出法（hold-out）按 8 : 2 划分，并通过 "
              "stratify 参数实施分层采样，使训练集与测试集中正负样本的比例与原始数据保持"
              "一致，避免因随机划分造成的类别比例失衡。留出法只做一次划分，而 k 折交叉"
              "验证会轮流留出一折作测试集并取 k 个结果的平均，二者的原理与分层采样的"
              "效果如下图（依据教材理论自绘）：")
    add_fig(doc, os.path.join(ASSETS, "concept", "fig_cross_validation.jpg"),
            "评估方法：10 折交叉验证与分层留出法（自绘）")

    label(doc, "【运行结果】")
    body(doc, "运行 data.info() 得到数据的结构与类型信息：")
    out_box(doc, nb_out_text(6))
    add_fig(doc, nb_out_image(7), "Simmons 数据集散点图（notebook 代码运行结果，白色主题）")
    body(doc, "散点图显示：Card 取值为 0/1 的两行样本在 Spending 方向上高度重叠，"
              "两类样本并不存在一条能够完全分开它们的直线，这预示了后续分类准确率不会很高。")

    # ---- c
    h3(doc, "c. 训练 Logistic 回归模型并输出分类准确率")
    req_box(doc, "c.训练Logistic回归模型，并输出最终的分类准确率。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(9))
    body(doc, "以 Spending 与 Card 两个特征构造特征矩阵 X，以 Coupon 为标签 y，"
              "随后划分训练集与测试集并训练模型：")
    code_box(doc, nb_code(10))
    body(doc, "在测试集上进行预测并计算准确率：")
    code_box(doc, nb_code(11))
    code_box(doc, nb_code(12))

    label(doc, "【原理与推导】")
    body(doc, "Logistic 回归的参数 w、b 采用极大似然估计求解。将 σ(z_i) 记作样本 i 属于"
              "正类的预测概率 p̂_i，则整个训练集的似然函数为：")
    e5 = add_eq(doc, "f_likelihood")
    body(doc, f"最大化式 {e5} 等价于最小化其相反数，即对数损失（log-loss），"
              "它是 Logistic 回归的优化目标：")
    e6 = add_eq(doc, "f_logloss")
    body(doc, f"式 {e6} 是一个关于 w、b 的凸函数，因此不存在局部最优解的困扰。"
              f"结合式 {e4}（Sigmoid 的导数性质）对式 {e6} 求偏导，可得极为简洁的梯度形式：")
    e7 = add_eq(doc, "f_grad")
    body(doc, f"式 {e7} 表明梯度只与“预测概率与真实标记之差”有关，与线性回归的残差形式"
              "完全一致。由于目标函数二阶可导，除梯度下降外还可使用收敛更快的牛顿法：")
    e8 = add_eq(doc, "f_newton")
    body(doc, f"式 {e8} 中的 H 为海森矩阵；scikit-learn 默认采用 L-BFGS 等拟牛顿方法求解。"
              "下图给出两类样本的对数损失曲线，可以看出：当预测方向错误时（y = 1 而 z < 0，"
              "或 y = 0 而 z > 0），损失随 |z| 近似线性增长，模型受到“指数级”的惩罚，"
              "这正是极大似然估计的统计学含义。")
    add_fig(doc, os.path.join(ASSETS, "theory", "fig_logloss.jpg"),
            "对数损失函数的凸性与惩罚机制（依据教材理论自绘）")
    body(doc, "模型训练完成后，用测试集上的分类准确率衡量性能。错误率与精度的定义为：")
    e9 = add_eq(doc, "f_acc")
    body(doc, f"式 {e9} 中 I(·) 为指示函数，取值 1 表示判断成立。精度与错误率之和为 1，"
              "两者是最常用但也是最粗糙的性能度量——当类别不平衡时，它无法反映模型"
              "对少数类的识别能力，因此后续 LDA 部分还补充了混淆矩阵、查准率、查全率"
              "与 F1 分数等指标。")

    label(doc, "【运行结果】")
    body(doc, f"测试集共 {int(0.2 * 100)} 个样本，模型输出的分类准确率为 60.00%：")
    out_box(doc, nb_out_text(12))
    body(doc, "该结果与 b 中散点图的观察结论一致：两个特征对 Coupon 的区分能力有限，"
              "且样本量仅 100 条，故线性模型难以取得更高精度。这一现象说明："
              "在评估模型之前，应先通过可视化与数据探查判断数据本身的可分性上限。")

    # ---- d
    h3(doc, "d. 绘制 Logistic 回归模型的决策边界")
    req_box(doc, "d.绘制Logistic回归模型的决策边界。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(14))
    body(doc, "由模型参数解出决策边界并绘图：")
    code_box(doc, nb_code(15))

    label(doc, "【原理与推导】")
    body(doc, "由式 " + e2.strip("()") + " 可知 σ(z) 关于 z = 0 处取值为 0.5。因此模型"
              "判定为正类与负类的分界恰好发生在 z = 0 处，即决策边界满足：")
    e10 = add_eq(doc, "f_boundary")
    body(doc, f"式 {e10} 说明：Logistic 回归的决策边界是由权向量 w 与偏置 b 确定的超平面，"
              "在二维特征空间中它退化为一条直线，w 是这条直线的法向量，其方向决定边界"
              "的朝向，b 则决定边界相对原点的偏移。代码中正是按 z = β₀ + β₁·Spending + "
              "β₂·Card = 0 反解出 Card 关于 Spending 的直线方程 Card = −(β₀ + β₁·Spending) / β₂。")
    body(doc, "为进一步说明该边界的几何意义与概率解释，下图以本实验真实数据重新拟合并"
              "绘制了决策边界及两侧的概率区域（背景颜色深浅代表预测概率）：")
    add_fig(doc, os.path.join(ASSETS, "theory", "fig_boundary_geo.jpg"),
            "Logistic 回归决策边界的几何解释与概率区域（Simmons 数据实测，自绘）")
    body(doc, "值得注意的是，决策边界默认取在后验概率为 0.5 处，这隐含了“两类错误代价"
              "相同”的假设。若实际任务中两类错误的代价不同，可通过移动分类阈值来调整边界，"
              "其综合评价手段即为 P-R 曲线与 ROC 曲线：ROC 曲线以假正例率为横轴、真正例率"
              "为纵轴，曲线下面积（AUC）可用来比较不同模型或不同阈值下的排序能力，"
              "其估算方法与原理如下图：")
    add_fig(doc, os.path.join(ASSETS, "concept", "fig_roc_auc.jpg"),
            "ROC 曲线与 AUC（左：本实验数据实测；右：特征选择对排序能力的影响）（自绘）")

    label(doc, "【运行结果】")
    add_fig(doc, os.path.join(ASSETS, "concept", "fig_boundary_result.jpg"),
            "决策边界的运行结果：散点、边界与误分类样本（自绘）")
    body(doc, "该图在决策边界的基础上叠加了训练集与测试集散点，并用红圈标出被误判的"
              "测试样本：测试集 20 个样本中错分 8 个，准确率 60.00%，与 c 中输出的数值"
              "完全一致。可以看出误分类样本集中在两类重叠的区域（Card = 0 而消费额较高、"
              "Card = 1 而消费额较低），这正是线性边界无法完全分开两类样本的直接体现。")

    # ------------------------------------------------------------------ 二
    doc.add_page_break()
    h1(doc, "二、线性判别分析")

    h2(doc, "1.实验目标")
    body(doc, "a.理解线性判别分析(LDA)的原理及其应用。")
    body(doc, "b.学习如何使用LDA进行分类，并比较其与其他分类算法的性能。")

    h2(doc, "2.实验数据")
    body(doc, "本实验使用鸢尾花 IRIS 数据集（直接加载使用 sklearn 库中的 datasets 引用"
              "鸢尾花数据集），该数据集包含 150 个样本，分为 3 种鸢尾花，每种花的特征"
              "包含萼片长度、萼片宽度、花瓣长度和花瓣宽度。")

    h2(doc, "3.实验要求")
    for s in ["a.编写Python代码实现线性判别分析。",
              "b.加载鸢尾花数据集，并对数据进行预处理和可视化。",
              "c.使用LDA进行分类，并输出模型的准确率、混淆均值等评估指标。",
              "d.将LDA投影结果可视化，观察不同类型的分布情况。"]:
        body(doc, s)

    h2(doc, "4、实验步骤")

    # ---- 2a
    h3(doc, "a. 实现线性判别分析")
    req_box(doc, "a.编写Python代码实现线性判别分析。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(18))

    label(doc, "【原理与推导】")
    body(doc, "线性判别分析的核心思想是“投影”：把高维样本投影到一条直线 y = wᵗx 上，"
              "并使投影结果满足——同类样本的投影点尽可能聚集，异类样本的投影点尽可能"
              "远离。为定量描述这一目标，先定义各类的均值向量与全体样本均值：")
    e11 = add_eq(doc, "f_class_mean")
    body(doc, "进而定义类内散度矩阵 S_w（衡量同类样本的分散程度）与类间散度矩阵 S_b"
              "（衡量异类均值之间的距离）：")
    e12 = add_eq(doc, "f_sw")
    e13 = add_eq(doc, "f_sb")
    body(doc, f"其中式 {e12} 为各类协方差矩阵之和，式 {e13} 为类均值相对全体均值的"
              "加权散布。投影后，同类样本的方差为 wᵗS_w w，异类均值之差为 wᵗ(μ₀ − μ₁)。"
              "于是 Fisher 提出用二者的比值作为优化准则：")
    e14 = add_eq(doc, "f_fisher")
    body(doc, f"式 {e14} 即 Fisher 准则，它是一个广义瑞利商。令 J(w) 对 w 求导并令其为零，"
              "可得最优投影方向所满足的方程，推导过程为：")
    e15 = add_eq(doc, "f_fisher_proof")
    body(doc, f"注意到式 {e15} 中 S_b w 的方向恒为 μ₀ − μ₁（因为 S_b = (μ₀ − μ₁)(μ₀ − μ₁)ᵗ），"
              "代入并令 λ = wᵗS_b w（它只影响 w 的长度而不影响方向），即得闭式解：")
    e16 = add_eq(doc, "f_fisher_sol")
    body(doc, f"式 {e16} 表明：最优投影方向只取决于两类均值之差经类内散度矩阵“白化”"
              "之后的朝向，与样本的具体分布形态（协方差）密切相关。下图以鸢尾花数据"
              "演示了该方向与两类样本分布的几何关系。")
    add_fig(doc, os.path.join(ASSETS, "theory", "fig_lda_geo.jpg"),
            "Fisher 判别准则的几何解释：类内散度椭圆与最优投影方向（自绘）")
    body(doc, "此外，sklearn 的 LinearDiscriminantAnalysis 是从同方差高斯假设出发，"
              "利用贝叶斯公式推导判别函数并取对数得到的；可以证明其解与式 "
              + e16.strip("()") + " 等价（仅相差一个与数据无关的尺度因子），"
              "同时该实现天然支持多分类推广，这正是本实验可以直接求解三分类问题的原因。")

    label(doc, "【运行结果】")
    body(doc, "该单元格仅实例化 LDA 模型对象，尚未训练，因此没有输出结果；"
              "模型的拟合与预测在 c 中完成。")

    # ---- 2b
    h3(doc, "b. 加载鸢尾花数据集并进行预处理与可视化")
    req_box(doc, "b.加载鸢尾花数据集，并对数据进行预处理和可视化。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(20))
    body(doc, "对特征进行标准化处理（标准化参数仅在训练集上估计）：")
    code_box(doc, nb_code(21))
    body(doc, "绘制花瓣长度与花瓣宽度的二维散点图，观察三类的分布：")
    code_box(doc, nb_code(22))

    label(doc, "【原理与推导】")
    body(doc, "鸢尾花数据集共 150 个样本、4 个连续特征、3 个类别，每类 50 个样本。"
              "代码按 8 : 2 划分训练集与测试集，并同样启用 stratify 分层采样，"
              "保证 120 个训练样本与 30 个测试样本中三类比例均为 1 : 1 : 1。")
    body(doc, "标准化（Z-score 规范化）将各特征变换为均值 0、方差 1 的分布：")
    e17 = add_eq(doc, "f_scale")
    body(doc, f"式 {e17} 中 μ、σ 分别为训练集上该特征的均值与标准差。对 LDA 而言，"
              "标准化并非可选步骤：求解投影方向需要计算 S_w 的逆矩阵，若各特征量纲"
              "相差悬殊（例如萼片长度以厘米计、而某特征数值很大），S_w 的条件数会急剧"
              "变差，投影方向将被量纲较大的特征主导。标准化后各特征处于同一尺度，"
              "S_w 的数值形态更稳定，Fisher 准则才能公平地权衡所有特征。")
    body(doc, "还需强调：标准化的 μ、σ 必须只在训练集上估计（fit），再变换（transform）"
              "测试集，否则测试集的信息会泄漏到训练过程中，导致评估结果虚高。")

    label(doc, "【运行结果】")
    add_fig(doc, nb_out_image(22), "鸢尾花数据集（花瓣长度—花瓣宽度）散点图（notebook 代码运行结果，白色主题）")
    body(doc, "从图中可见：山鸢尾（setosa）与另外两类在“花瓣长度”方向上几乎完全分离，"
              "而变色鸢尾（versicolor）与维吉尼亚鸢尾（virginica）在所选的两个特征上"
              "存在少量重叠，这为 c 中“三分类是否会被完全正确区分”提供了直观预期。")

    # ---- 2c
    h3(doc, "c. 使用 LDA 进行分类并输出评估指标")
    req_box(doc, "c.使用LDA进行分类，并输出模型的准确率、混淆均值等评估指标。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(24))
    code_box(doc, nb_code(25))
    code_box(doc, nb_code(26))

    label(doc, "【原理与推导】")
    body(doc, "LDA 在完成投影方向的求解后，对样本的判别方式是：先假设各类服从协方差"
              "相同的高斯分布，再由贝叶斯公式写出后验概率并取对数，得到第 k 类的判别函数：")
    e18 = add_eq(doc, "f_lda_disc")
    body(doc, f"式 {e18} 中 π_k 为先验概率，Σ 为各类共享的协方差矩阵。样本被划入判别"
              "函数取值最大的那一类。由于各类判别函数关于 x 都是线性的，任意两类之间的"
              "决策面依然是一个超平面：")
    e19 = add_eq(doc, "f_lda_binary_disc")
    body(doc, f"式 {e19} 说明了 LDA 与 Logistic 回归在形式上的统一性：二者都产生线性"
              "决策边界，区别在于参数估计的准则不同——前者基于高斯分布假设与 Fisher 准则，"
              "后者基于条件似然的极大化。")
    body(doc, "对于多分类问题，LDA 通过求解广义特征值问题得到多个投影方向，最多保留 "
              "K − 1 个（K 为类别数），从而实现有监督降维：")
    e20 = add_eq(doc, "f_lda_multi")
    body(doc, f"式 {e20} 解释了 d 中投影结果维度的由来：3 个类别最多保留 2 个判别方向。")
    body(doc, "在评估环节，除准确率外还输出混淆矩阵、查准率、查全率与 F1 分数。"
              "这些指标由混淆矩阵中的 TP、FP、FN、TN 四个计数导出：")
    e21 = add_eq(doc, "f_prf")
    body(doc, f"式 {e21} 中：查准率 P 衡量“被判为正例的样本中真正例的比例”（不误报），"
              "查全率 R 衡量“真正的正例中被找出来的比例”（不漏报），F1 为二者的调和平均，"
              "用于在查准率与查全率不可兼得时进行综合评价。混淆矩阵与查准率、查全率的"
              "定义如下图：")
    add_fig(doc, os.path.join(ASSETS, "concept", "fig_confusion_matrix.jpg"),
            "混淆矩阵的构成与由它导出的查准率、查全率、F1（自绘）")
    body(doc, "当模型输出概率并需要比较不同阈值下的表现时，可进一步使用 P-R 曲线。"
              "P-R 曲线的平衡点（查准率 = 查全率）常用于度量曲线交叉时分类器的性能高低：")
    add_fig(doc, os.path.join(ASSETS, "concept", "fig_pr_curve.jpg"),
            "P-R 曲线与平衡点（左：原理示意；右：本实验数据实测）（自绘）")

    label(doc, "【运行结果】")
    body(doc, "标准化后的特征使 LDA 在测试集上取得了 100% 的分类准确率：")
    out_box(doc, nb_out_text(24))
    body(doc, "混淆矩阵为对角阵，说明 30 个测试样本全部被判对（每一类的 10 个测试样本"
              "都正确落入对角线）：")
    out_box(doc, nb_out_text(25))
    body(doc, "分类报告给出了每一类的查准率、查全率与 F1 分数，三类均为 1.00；"
              "macro avg 与 weighted avg 也均为 1.00，说明模型在三类上的表现均衡，"
              "不存在偏向某一类的情况：")
    out_box(doc, nb_out_text(26))
    body(doc, "该结果优于第一部分 Logistic 回归在 Simmons 数据集上的 60% 准确率，"
              "原因有两点：一是鸢尾花数据本身在特征空间中可分性较好；"
              "二是 LDA 利用了类内散度矩阵的全局信息，属于生成式模型，"
              "在小样本条件下往往比判别式的 Logistic 回归更稳健。")

    # ---- 2d
    h3(doc, "d. 将 LDA 投影结果可视化")
    req_box(doc, "d.将LDA投影结果可视化，观察不同类型的分布情况。")
    label(doc, "【代码实现】")
    code_box(doc, nb_code(28))
    code_box(doc, nb_code(29))

    label(doc, "【原理与推导】")
    body(doc, f"投影变换即 y = wᵗx，其方向由式 {e16} 确定；对多分类问题则取广义特征值"
              f"问题的前 K − 1 个特征向量构成投影矩阵 V，如式 {e20}。投影后的坐标"
              "（Linear Discriminant 1、2）是原始 4 维特征在“类间散度最大、类内散度最小”"
              "意义下的最优二维表示，因此投影图中同类样本应当彼此聚集、不同类样本应当"
              "彼此分离。检验投影图是否满足这一形态，等价于对 Fisher 准则优化目标的"
              "一次直观验证。")

    label(doc, "【运行结果】")
    body(doc, "训练集 120 个样本、3 个类别，投影后保留 2 个判别方向，故输出形状为 (120, 2)：")
    out_box(doc, nb_out_text(28))
    add_fig(doc, nb_out_image(29), "LDA 投影结果（Linear Discriminant 1—2）（notebook 代码运行结果，白色主题）")
    body(doc, "投影图中三类样本各自聚集成簇、簇间几乎不重叠，与 c 中 100% 的准确率"
              "相互印证，说明 LDA 找到的判别方向确实实现了“类内紧凑、类间疏远”的目标。")

    # ------------------------------------------------------------------ 结尾
    para(doc, "实验结束后下周六晚 10 点前需要提交电子版实验报告。文件命名："
              "MLSY0x0x学号姓名.doc。其中，第一个 x 表示第 x 次实验，如第一次实验为 01，"
              "第二个 x 表示班次，如 01 表示 1 班。文件中内插图片应为 jpg 格式。"
              "书面实验报告在所有实验完成后将四次实验按要求汇总打印后统一交给助教陈天云，"
              "具体时间另行通知。", size=10.5, before=12, after=0, indent_chars=2)

    doc.save(OUT_PATH)
    print("已生成：", OUT_PATH)
    print(f"插图 {NUM.fig} 张，公式 {NUM.eq} 个")


if __name__ == "__main__":
    build()
