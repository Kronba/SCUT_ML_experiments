# -*- coding: utf-8 -*-
"""实验三素材：贝叶斯分类器与 BP 算法的理论原理图 + 真实数据图 + 公式。

统一 report_style 主题；实测类插图全部使用 data/exp3 下的真实数据，
并与 notebook 的划分方式/随机种子保持一致。
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

import report_style as rs

OUT = os.path.join(rs.ASSETS, "exp3", "theory")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED
RS = 42


# ---------------------------------------------------------------------------
# 数据读取（与 notebook 完全一致）
# ---------------------------------------------------------------------------
def load_emails():
    emails, labels = [], []
    for path, lab in ((os.path.join(rs.PROJ_DIR, "data/exp3/email/ham/*.txt"), 0),
                      (os.path.join(rs.PROJ_DIR, "data/exp3/email/spam/*.txt"), 1)):
        for f in sorted(glob.glob(path)):
            with open(f, encoding="utf-8", errors="ignore") as fh:
                emails.append(fh.read())
                labels.append(lab)
    df = pd.DataFrame({"text": emails, "label": labels})
    df["text"] = df["text"].str.lower()
    return df


def load_wine():
    df = pd.read_csv(os.path.join(rs.PROJ_DIR, "data/exp3/winequality-red.csv"), sep=";")
    df["grade"] = df["quality"].apply(lambda x: "bad" if x <= 5 else "good")
    df["label"] = df["grade"].map({"bad": 0, "good": 1})
    return df


WINE_FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                 "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                 "pH", "sulphates", "alcohol"]


# ---------------------------------------------------------------------------
# 一、贝叶斯分类器
# ---------------------------------------------------------------------------
def fig_bayes_principle():
    """贝叶斯决策的四个要素 + 朴素条件独立假设。"""
    rs.apply()
    fig = plt.figure(figsize=(9.8, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.12)

    # (a) 先验 / 似然 / 后验 的关系
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.text(5, 6.6, "（a）贝叶斯决策公式", ha="center", fontsize=11.5)

    def box(x, y, w, h, text, fc, ec="#8a8a8a", fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    box(0.3, 4.6, 4.2, 1.35, "先验概率 $P(c)$\n各类邮件出现的比例", "#eef3f9")
    box(5.5, 4.6, 4.2, 1.35, "类条件概率 $P(\\mathbf{x}\\,|\\,c)$\n该类别下出现这封邮件的可能", "#fae3d0")
    box(2.9, 2.4, 4.2, 1.35, "后验概率\n$P(c\\,|\\,\\mathbf{x})\\propto P(c)P(\\mathbf{x}\\,|\\,c)$", "#dce9f6", fs=10.5)
    ax.annotate("", xy=(4.6, 3.75), xytext=(2.4, 4.6),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))
    ax.annotate("", xy=(5.4, 3.75), xytext=(7.6, 4.6),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))
    ax.text(5.0, 1.7, "判决规则：把邮件判给后验概率最大的类别", ha="center", fontsize=10.5, color="#1F4E79")
    ax.text(5.0, 0.9, "（分母 $P(\\mathbf{x})$ 与类别无关，可略去）", ha="center", fontsize=9.5, color="#666666")

    # (b) 朴素条件独立假设
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.text(5, 6.6, "（b）“朴素”条件独立假设", ha="center", fontsize=11.5)
    box(3.6, 4.9, 2.8, 1.1, "类别 $c$", "#dce9f6", fs=10.5)
    words = ["词 $x_1$", "词 $x_2$", "词 $x_3$", "$\\cdots$", "词 $x_d$"]
    xs = np.linspace(0.55, 8.6, 5)
    for x, w in zip(xs, words):
        box(x, 2.2, 1.6, 1.0, w, "#f7f7f7", fs=10)
        ax.annotate("", xy=(x + 0.8, 3.2), xytext=(5.0, 4.9),
                    arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=0.9))
    ax.text(5.0, 1.2, "各特征在给定类别下相互独立：$P(\\mathbf{x}\\,|\\,c)=\\prod_{i=1}^{d}P(x_i\\,|\\,c)$",
            ha="center", fontsize=10.5)
    ax.text(5.0, 0.5, "该假设显然不严格成立，但使参数估计量从指数级降到线性级",
            ha="center", fontsize=9.5, color="#666666")

    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_bayes_principle.jpg"))


def fig_laplace_smoothing():
    """拉普拉斯平滑：解决未出现词导致的零概率问题。"""
    rs.apply()
    words = ["free", "click", "meeting", "project", "viagra", "offer"]
    raw = np.array([0.035, 0.028, 0.041, 0.030, 0.000, 0.000])
    lap = (raw * 200 + 1) / (200 + 1200)      # 示意：+(α)/(N+αV)
    x = np.arange(len(words))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.2, raw, width=0.4, color=ORANGE, label="极大似然估计（未平滑）")
    ax.bar(x + 0.2, lap, width=0.4, color=BLUE, label=r"拉普拉斯平滑（$\alpha=1$）")
    for xi, v in zip(x, raw):
        if v == 0:
            ax.annotate("$=0$", xy=(xi - 0.2, 0.0005), xytext=(xi - 0.2, 0.0035),
                        ha="center", fontsize=9, color=RED)
    ax.set_ylim(0, 0.055)
    ax.text(0.5, 0.975, "某个词若在某一类训练样本中从未出现，其条件概率被估为 0，"
                        "连乘后整封邮件的后验概率也会变为 0",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=RED,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdf3ee", edgecolor=RED, lw=0.8))
    ax.set_xticks(x, words)
    ax.set_xlabel("词汇")
    ax.set_ylabel("$P(\\mathrm{word}\\,|\\,c)$")
    ax.set_title("零概率问题与拉普拉斯平滑")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig3_laplace_smoothing.jpg"))


def fig_email_overview():
    """邮件数据集：类别分布与文本长度分布。"""
    rs.apply()
    df = load_emails()
    lengths = df["text"].str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax = axes[0]
    vc = df["label"].value_counts().sort_index()
    bars = ax.bar(["Ham（正常邮件）", "Spam（垃圾邮件）"], vc.values, color=[ORANGE, BLUE],
                  width=0.5, edgecolor="white")
    for r, v in zip(bars, vc.values):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.4, f"{v} 封", ha="center", fontsize=10)
    ax.set_ylim(0, 29)
    ax.set_ylabel("样本数")
    ax.set_title("（a）两类邮件数量完全均衡")
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    bins = np.arange(0, lengths.max() + 20, 15)
    ax.hist(lengths[df["label"] == 0], bins=bins, color=ORANGE, alpha=0.8, label="Ham")
    ax.hist(lengths[df["label"] == 1], bins=bins, color=BLUE, alpha=0.8, label="Spam")
    ax.set_xlabel("邮件词数")
    ax.set_ylabel("样本数")
    ax.set_title("（b）两类邮件的长度分布接近")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_email_overview.jpg"))


def fig_cv_scores():
    """5 折交叉验证的每折准确率（真实结果）。"""
    rs.apply()
    folds = ["第1折", "第2折", "第3折", "第4折", "第5折"]
    scores = [100.0, 100.0, 90.0, 80.0, 100.0]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(folds, scores, color=BLUE, width=0.55, edgecolor="white")
    bars[int(np.argmin(scores))].set_color(RED)   # 标出最差的一折
    for r, v in zip(bars, scores):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10)
    mean = float(np.mean(scores))
    ax.axhline(mean, color=GREEN, ls="--", lw=1.6, label=f"平均准确率 {mean:.2f}%")
    ax.set_ylim(0, 112)
    ax.set_ylabel("准确率 / %")
    ax.set_title("5 折交叉验证结果（仅 50 条样本，折间波动明显）")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig3_cv_scores.jpg"))


def fig_top_words():
    """朴素贝叶斯学到的判别性最强的词（真实模型参数）。"""
    rs.apply()
    df = load_emails()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=RS, stratify=df["label"])
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    Xtr = vec.fit_transform(X_train)
    nb = MultinomialNB().fit(Xtr, y_train)
    vocab = np.array(vec.get_feature_names_out())
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]      # spam 相对 ham 的对数几率
    order = np.argsort(diff)
    top_spam = order[-10:][::-1]
    top_ham = order[:10][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax = axes[0]
    ax.barh(vocab[top_spam][::-1], diff[top_spam][::-1], color=BLUE, height=0.62)
    ax.set_xlabel(r"$\log P(w\,|\,\mathrm{Spam})-\log P(w\,|\,\mathrm{Ham})$")
    ax.set_title("（a）最倾向于判为垃圾邮件的词")
    ax.grid(axis="x", alpha=0.35)
    ax = axes[1]
    ax.barh(vocab[top_ham][::-1], diff[top_ham][::-1], color=ORANGE, height=0.62)
    ax.set_xlabel(r"$\log P(w\,|\,\mathrm{Spam})-\log P(w\,|\,\mathrm{Ham})$")
    ax.set_title("（b）最倾向于判为正常邮件的词")
    ax.grid(axis="x", alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_top_words.jpg"))


# ---------------------------------------------------------------------------
# 二、BP 算法
# ---------------------------------------------------------------------------
def fig_neuron_activation():
    """神经元结构与常用激活函数。"""
    rs.apply()
    fig = plt.figure(figsize=(9.8, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.text(5, 6.6, "（a）单个神经元", ha="center", fontsize=11.5)
    ins = [("$x_1$", 5.4), ("$x_2$", 4.2), ("$\\vdots$", 3.4), ("$x_d$", 2.5)]
    for name, y in ins:
        ax.add_patch(Circle((1.4, y), 0.42, facecolor="#f7f7f7", edgecolor="#8a8a8a"))
        ax.text(1.4, y, name, ha="center", va="center", fontsize=10)
        ax.annotate("", xy=(3.9, 3.95), xytext=(1.85, y),
                    arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=0.9))
    ax.add_patch(Circle((4.6, 3.95), 0.75, facecolor="#dce9f6", edgecolor="#4a7fb5", lw=1.4))
    ax.text(4.6, 3.95, "$\\sum$", ha="center", va="center", fontsize=14)
    ax.annotate("", xy=(6.9, 3.95), xytext=(5.4, 3.95),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3))
    ax.text(6.15, 4.35, "$f(\\cdot)$", fontsize=10.5)
    ax.add_patch(Circle((7.6, 3.95), 0.62, facecolor="#fae3d0", edgecolor="#c07b45", lw=1.3))
    ax.text(7.6, 3.95, "$a$", ha="center", va="center", fontsize=11.5)
    ax.annotate("", xy=(8.9, 3.95), xytext=(8.25, 3.95),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3))
    ax.text(4.6, 1.5, "$a=f\\left(\\sum_i w_i x_i+b\\right)$", ha="center", fontsize=11.5)

    ax = fig.add_subplot(gs[0, 1])
    z = np.linspace(-5, 5, 400)
    ax.plot(z, np.maximum(0, z), color=BLUE, lw=2, label=r"ReLU: $\max(0,z)$")
    ax.plot(z, 1 / (1 + np.exp(-z)), color=ORANGE, lw=2, ls="--", label=r"Sigmoid: $\frac{1}{1+e^{-z}}$")
    ax.plot(z, np.tanh(z), color=GREEN, lw=2, ls="-.", label=r"Tanh: $\tanh z$")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$f(z)$")
    ax.set_title("（b）常用激活函数")
    ax.set_ylim(-1.6, 5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_neuron_activation.jpg"))


def fig_bp_flow():
    """前向传播与误差反向传播示意。"""
    rs.apply()
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    layers = [(1.6, 4, "输入层\n11 个特征"), (5.0, 2, "隐藏层\n$n$ 个神经元"), (8.6, 2, "输出层\n2 类")]

    def nodes(cx, n, spread=2.0, color="#dce9f6"):
        ys = np.linspace(3 - spread / 2, 3 + spread / 2, n) if n > 1 else [3.0]
        for y in ys:
            ax.add_patch(Circle((cx, y), 0.26, facecolor=color, edgecolor="#6a8fb5", lw=1.0))
        return ys

    y0 = nodes(layers[0][0], 4, 2.6)
    y1 = nodes(layers[1][0], 3, 2.0)
    y2 = nodes(layers[2][0], 2, 1.2, "#fae3d0")
    for a in y0:
        for b in y1:
            ax.annotate("", xy=(layers[1][0] - 0.26, b), xytext=(layers[0][0] + 0.26, a),
                        arrowprops=dict(arrowstyle="-|>", color="#b9c9d8", lw=0.7))
    for a in y1:
        for b in y2:
            ax.annotate("", xy=(layers[2][0] - 0.26, b), xytext=(layers[1][0] + 0.26, a),
                        arrowprops=dict(arrowstyle="-|>", color="#b9c9d8", lw=0.7))
    for cx, _, label in layers:
        ax.text(cx, 0.75, label, ha="center", fontsize=9.5)
    ax.annotate("", xy=(10.6, 3.0), xytext=(9.2, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))
    ax.text(10.75, 3.0, "损失 $L$", fontsize=10, va="center")
    # 反向传播
    ax.annotate("", xy=(1.6, 5.35), xytext=(10.0, 5.35),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                connectionstyle="arc3,rad=0"))
    ax.text(5.8, 5.6, "误差反向传播：逐层计算 $\\delta$，得到各层梯度",
            ha="center", fontsize=10.5, color=RED)
    ax.text(6.0, 1.6, "前向传播：$a^{(l)}=f\\left(W^{(l)}a^{(l-1)}+b^{(l)}\\right)$",
            ha="center", fontsize=10.5, color="#1F4E79")
    rs.save(fig, os.path.join(OUT, "fig3_bp_flow.jpg"))


def fig_quality_distribution():
    """红酒综合品质分布与 bad/good 划分。"""
    rs.apply()
    df = load_wine()
    vc = df["quality"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [ORANGE if q <= 5 else BLUE for q in vc.index]
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, width=0.6, edgecolor="white")
    for r, v in zip(bars, vc.values):
        ax.text(r.get_x() + r.get_width() / 2, v + 12, str(v), ha="center", fontsize=9.5)
    ax.axvline(2.5, color=RED, ls="--", lw=1.6)
    ax.text(0.9, 745, "bad（3~5 分）\n744 个样本", ha="center", fontsize=9.5, color=ORANGE)
    ax.text(4.1, 745, "good（6~8 分）\n855 个样本", ha="center", fontsize=9.5, color=BLUE)
    ax.set_xlabel("quality（综合品质得分）")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, 880)
    ax.set_title("红酒品质分布与二分类划分")
    ax.grid(axis="y", alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig3_quality_distribution.jpg"))


def fig_scaling_effect():
    """归一化前后的特征量纲对比。"""
    rs.apply()
    df = load_wine()
    X = df[WINE_FEATURES].values
    Xs = MinMaxScaler().fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=False)
    for ax, data, title in ((axes[0], X, "（a）归一化前：各特征量纲相差数十倍"),
                            (axes[1], Xs, "（b）归一化后：全部压缩到 [0, 1]")):
        bp = ax.boxplot([data[:, i] for i in range(data.shape[1])], patch_artist=True,
                        widths=0.6, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#dce9f6")
            patch.set_edgecolor("#5b87b5")
        for med in bp["medians"]:
            med.set_color(RED)
            med.set_linewidth(1.3)
        ax.set_xticks(range(1, 12), [f[:9] for f in WINE_FEATURES], rotation=45,
                      ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.35)
        if data is X:
            ax.set_yscale("log")
            ax.set_ylabel("原始取值（对数坐标）")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_scaling_effect.jpg"))


def fig_loss_curves():
    """不同隐藏层规模的训练损失下降曲线（真实训练过程）。"""
    rs.apply()
    df = load_wine()
    X, y = df[WINE_FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RS, stratify=y)
    sc = MinMaxScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for n, color, ls in ((2, RED, "-"), (1, BLUE, "--"), (10, GREEN, "-.")):
        m = MLPClassifier(hidden_layer_sizes=(n,), activation="relu", solver="adam",
                          max_iter=1000, random_state=RS).fit(Xtr, y_train)
        ax.plot(m.loss_curve_, color=color, ls=ls, lw=1.8,
                label=f"隐藏层 {n} 个神经元（测试误差率 "
                      f"{(1 - accuracy_score(y_test, m.predict(Xte))) * 100:.2f}%）")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("训练损失")
    ax.set_title("不同隐藏层规模的训练损失下降过程")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig3_loss_curves.jpg"))


def fig_hidden_sweep():
    """隐藏层神经元个数与测试误差率（真实结果）。"""
    rs.apply()
    ns = [1, 2, 3, 5, 10, 15]
    err = [26.0417, 54.1667, 26.0417, 26.6667, 26.6667, 26.2500]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    colors = [GREEN if e == min(err) else (RED if n == 2 else BLUE) for n, e in zip(ns, err)]
    bars = ax.bar([str(n) for n in ns], err, color=colors, width=0.55, edgecolor="white")
    for r, v, n in zip(bars, err, ns):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.0, f"{v:.2f}%", ha="center", fontsize=9.5)
    ax.annotate("题目指定的 2 个神经元\n恰好落在最差解", xy=(1, 54.17), xytext=(1.55, 46),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.axhline(min(err), color=GREEN, ls="--", lw=1.3)
    ax.text(5.4, min(err) + 1.6, f"最优 {min(err):.2f}%（1 个神经元）", color=GREEN, fontsize=9.5)
    ax.set_xlabel("隐藏层神经元个数")
    ax.set_ylabel("测试错误率 / %")
    ax.set_ylim(0, 62)
    ax.set_title("隐藏层神经元个数对测试错误率的影响")
    ax.grid(axis="y", alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig3_hidden_sweep.jpg"))


# ===========================================================================
# 公式
# ===========================================================================
FORMULAS = {
    # ---- 贝叶斯分类器 ----
    "f3_bayes": r"$P\left(c\,|\,\mathbf{x}\right)=\dfrac{P\left(c\right)P\left(\mathbf{x}\,|\,c\right)}{P\left(\mathbf{x}\right)}$",
    "f3_argmax": r"$\hat{y}=\arg\max_{c\in\mathcal{Y}}\;P\left(c\right)\prod_{i=1}^{d}P\left(x_i\,|\,c\right)$",
    "f3_prior": r"$\hat{P}\left(c\right)=\dfrac{N_c}{N}$",
    "f3_mle": r"$\hat{P}\left(x_i\,|\,c\right)=\dfrac{N_{ic}}{N_c}$",
    "f3_laplace": r"$\hat{P}\left(x_i\,|\,c\right)=\dfrac{N_{ic}+\alpha}{N_c+\alpha V},\qquad \alpha>0$",
    "f3_logpost": r"$\log P\left(c\,|\,\mathbf{x}\right)\propto\log \hat{P}\left(c\right)+\sum_{i=1}^{d}\log \hat{P}\left(x_i\,|\,c\right)$",
    "f3_tfidf": r"$\mathrm{tfidf}\left(t,d\right)=\mathrm{tf}\left(t,d\right)\times\log\dfrac{N}{\mathrm{df}\left(t\right)}$",
    "f3_odds": r"$\mathrm{score}\left(w\right)=\log P\left(w\,|\,\mathrm{Spam}\right)-\log P\left(w\,|\,\mathrm{Ham}\right)$",
    "f3_cv": r"$acc_{\mathrm{CV}}=\dfrac{1}{K}\sum_{k=1}^{K}acc_k$",
    # ---- BP 算法 ----
    "f3_neuron": r"$a=f\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}+b\right)$",
    "f3_relu": r"$\mathrm{ReLU}(z)=\max\left(0,z\right)$",
    "f3_sigmoid": r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$",
    "f3_forward": r"$\mathbf{z}^{(l)}=W^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)},\qquad \mathbf{a}^{(l)}=f\left(\mathbf{z}^{(l)}\right)$",
    "f3_loss": r"$L=-\dfrac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_{ik}\log\hat{p}_{ik}$",
    "f3_delta_out": r"$\boldsymbol{\delta}^{(L)}=\nabla_{\mathbf{a}}L\odot f'\left(\mathbf{z}^{(L)}\right)$",
    "f3_delta_hidden": r"$\boldsymbol{\delta}^{(l)}=\left(\left(W^{(l+1)}\right)^{\mathrm{T}}\boldsymbol{\delta}^{(l+1)}\right)\odot f'\left(\mathbf{z}^{(l)}\right)$",
    "f3_grad": r"$\dfrac{\partial L}{\partial W^{(l)}}=\boldsymbol{\delta}^{(l)}\left(\mathbf{a}^{(l-1)}\right)^{\mathrm{T}},\qquad \dfrac{\partial L}{\partial \mathbf{b}^{(l)}}=\boldsymbol{\delta}^{(l)}$",
    "f3_update": r"$W^{(l)}\leftarrow W^{(l)}-\eta\dfrac{\partial L}{\partial W^{(l)}},\qquad \mathbf{b}^{(l)}\leftarrow\mathbf{b}^{(l)}-\eta\dfrac{\partial L}{\partial \mathbf{b}^{(l)}}$",
    "f3_minmax": r"$x'=\dfrac{x-x_{\min}}{x_{\max}-x_{\min}}\in\left[0,1\right]$",
    "f3_split": r"$m_{\mathrm{train}}:m_{\mathrm{test}}=7:3$",
    "f3_error": r"$E=1-acc=\dfrac{1}{m}\sum_{i=1}^{m}\mathrm{I}\left(f\left(\mathbf{x}_i\right)\neq y_i\right)$",
    "f3_prf": r"$P=\dfrac{TP}{TP+FP},\qquad R=\dfrac{TP}{TP+FN},\qquad F1=\dfrac{2PR}{P+R}$",
}


def render_formulas():
    print("渲染公式 ...")
    rs.apply()
    out_dir = os.path.join(rs.ASSETS, "exp3", "formula")
    os.makedirs(out_dir, exist_ok=True)
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=20, color="black")
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300,
                    bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
    print(f"  共 {len(FORMULAS)} 个公式 -> 素材/exp3/formula/")


if __name__ == "__main__":
    print("绘制实验三理论原理图 ...")
    fig_bayes_principle()
    fig_laplace_smoothing()
    fig_email_overview()
    fig_cv_scores()
    fig_top_words()
    fig_neuron_activation()
    fig_bp_flow()
    fig_quality_distribution()
    fig_scaling_effect()
    fig_loss_curves()
    fig_hidden_sweep()
    render_formulas()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
