# -*- coding: utf-8 -*-
"""实验四素材：AdaBoost 集成学习与决策树（非均衡分类）的原理图、真实数据图与公式。

统一 report_style 主题（白底、中文衬线、Computer Modern、600 dpi）。
notebook 中的两张决策树图用 figsize=(20, 10) 绘制，缩到版心宽度后文字不可读，
因此这里用自绘的紧凑版树图（只展示前三层）替代。
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import report_style as rs

OUT = os.path.join(rs.ASSETS, "exp4", "theory")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED
RS = 42


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_florence():
    df = pd.read_excel(os.path.join(rs.PROJ_DIR, "data", "exp4", "Florence.xlsx"))
    return df


def split_florence(df, features=None):
    X = df[features] if features else df.drop(columns=["Seq#", "Florence"])
    y = df["Florence"]
    return train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)


def make_tree(features=None):
    return DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=10,
                                  min_samples_leaf=5, class_weight="balanced", random_state=RS)


# ---------------------------------------------------------------------------
# 自绘紧凑决策树
# ---------------------------------------------------------------------------
def draw_compact_tree(ax, est, feature_names, class_names, max_depth=3,
                      node_fs=9.0, leaf_colors=("#fae3d0", "#d9e6f5")):
    """把 sklearn 决策树画成紧凑版（节点只写划分条件与样本数），只画前 max_depth 层。"""
    tree = est.tree_
    n_node_samples = tree.n_node_samples
    value = tree.value
    total = value[0].sum()

    pos = {}

    def layout(node, depth, left, width):
        if depth >= max_depth or tree.children_left[node] == -1:
            x = left + width / 2
            pos[node] = (x, -depth)
            return x
        xl = layout(tree.children_left[node], depth + 1, left, width / 2)
        xr = layout(tree.children_right[node], depth + 1, left + width / 2, width / 2)
        pos[node] = ((xl + xr) / 2, -depth)
        return pos[node][0]

    layout(0, 0, 0.0, 1.0)

    # 连线
    for node, (x, y) in list(pos.items()):
        for child in (tree.children_left[node], tree.children_right[node]):
            if child in pos:
                cx, cy = pos[child]
                ax.plot([x, cx], [y - 0.16, cy + 0.16], color="#9aa5b1", lw=0.9, zorder=1)
    # 节点
    for node, (x, y) in pos.items():
        cnt = int(n_node_samples[node])
        vals = value[node][0]
        majority = int(np.argmax(vals))
        is_leaf = tree.children_left[node] == -1 or y == -max_depth
        if is_leaf or y == -max_depth:
            frac = vals[majority] / vals.sum()
            base = leaf_colors[majority]
            face = tuple(1 - (1 - c) * (0.45 + 0.55 * frac) for c in
                         matplotlib.colors.to_rgb(base))
            text = f"{class_names[majority]}\n$n$={cnt}"
        else:
            feat = feature_names[tree.feature[node]]
            thr = tree.threshold[node]
            face = "#f4f6f8"
            text = f"{feat} ≤ {thr:.2f}\n$n$={cnt}"
        ax.add_patch(FancyBboxPatch((x - 0.088, y - 0.13), 0.176, 0.26,
                                    boxstyle="round,pad=0.006", facecolor=face,
                                    edgecolor="#7a8899", linewidth=0.9, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=node_fs, zorder=3)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-max_depth - 0.32, 0.34)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# 一、AdaBoost 集成学习
# ---------------------------------------------------------------------------
def fig_adaboost_principle():
    rs.apply()
    fig = plt.figure(figsize=(9.8, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)

    # (a) 弱学习器加权组合
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.text(5, 6.6, "（a）弱学习器加权组合", ha="center", fontsize=11.5)

    def box(x, y, w, h, text, fc, fs=9.5, ec="#8a8a8a"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor=ec, linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    ys = [5.0, 3.5, 2.0]
    alphas = ["$\\alpha_1=0.9$", "$\\alpha_2=0.6$", "$\\alpha_3=1.4$"]
    for y, a, tag in zip(ys, alphas, ["$h_1$", "$h_2$", "$h_3$"]):
        box(0.4, y, 3.1, 1.0, f"决策树桩 {tag}", "#f4f6f8")
        ax.text(3.75, y + 0.5, a, fontsize=9.5, va="center", color="#1F4E79")
        ax.annotate("", xy=(6.6, 3.6), xytext=(4.35, y + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#9aa5b1", lw=1.0))
    box(6.6, 3.1, 3.0, 1.0, "加权投票\n$H(x)=\\mathrm{sign}\\sum_t\\alpha_th_t(x)$", "#dce9f6", fs=9)
    ax.annotate("", xy=(8.1, 2.55), xytext=(8.1, 3.1),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2))
    ax.text(8.1, 2.15, "最终判决", ha="center", fontsize=9.5)
    ax.text(5.0, 0.9, "每轮提高被上一轮判错样本的权重，\n使新的弱学习器专注于难分样本",
            ha="center", fontsize=9.5, color="#666666")

    # (b) α 与加权错误率的关系
    ax = fig.add_subplot(gs[0, 1])
    eps = np.linspace(0.001, 0.499, 400)
    alpha = 0.5 * np.log((1 - eps) / eps)
    ax.plot(eps, alpha, color=BLUE, lw=2)
    ax.axvline(0.5, color=RED, ls="--", lw=1.3)
    ax.plot(0.5, 0.0, "o", color=RED, ms=5)
    ax.annotate("$\\varepsilon=0.5$：与随机猜测无异，权重 $\\alpha=0$", xy=(0.497, 0.02),
                xytext=(0.045, 0.32), ha="left", fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9,
                                connectionstyle="arc3,rad=-0.15"))
    ax.set_xlabel("加权错误率 $\\varepsilon_t$")
    ax.set_ylabel("弱学习器权重 $\\alpha_t$")
    ax.set_title("（b）权重随错误率的变化")
    ax.set_xlim(0, 0.5)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig4_adaboost_principle.jpg"))


def fig_pipeline():
    rs.apply()
    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.set_axis_off()

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor="#8a8a8a", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, x1, y=1.5):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=11, color="#666666", lw=1.1))

    box(0.0, 0.95, 1.5, 1.1, "Credit_card\n+ label\n按 Ind_ID 合并", "#eef3f9", 8.5)
    arrow(1.55, 1.95)
    box(2.0, 0.95, 1.9, 1.1, "特征分流\n数值列 / 类别列", "#f4f6f8", 8.5)
    arrow(3.95, 4.35)
    box(4.4, 1.55, 2.0, 0.95, "数值列：中位数填充", "#fae3d0", 8.5)
    box(4.4, 0.35, 2.0, 0.95, "类别列：众数填充\n+ One-Hot 编码", "#fae3d0", 8.5)
    arrow(6.45, 6.85)
    box(6.9, 0.95, 1.9, 1.1, "分层划分\n7 : 3", "#f4f6f8", 8.5)
    arrow(8.85, 9.25)
    box(9.3, 0.95, 2.6, 1.1, "AdaBoost\n决策树桩 × 100", "#dce9f6", 9)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig4_pipeline.jpg"))


def fig_imbalance_baseline():
    rs.apply()
    acc_model = 0.8817204301075269 * 100
    acc_const = 412 / 465 * 100
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    ax = axes[0]
    bars = ax.bar(["Normal（正常交易）", "Fraud（欺诈交易）"], [1373, 175],
                  color=[ORANGE, BLUE], width=0.5, edgecolor="white")
    for r, v in zip(bars, [1373, 175]):
        ax.text(r.get_x() + r.get_width() / 2, v + 20, f"{v}\n({v / 1548 * 100:.2f}%)",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1600)
    ax.set_ylabel("样本数")
    ax.set_title("（a）信用卡交易数据的类别分布")
    ax.grid(axis="y", alpha=0.35)
    ax.annotate("欺诈样本仅占 11.30%\n正负样本比约 1 : 7.8", xy=(1, 175), xytext=(0.25, 900),
                fontsize=9.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))

    ax = axes[1]
    names = ["全部预测为 Normal\n（常数基线）", "AdaBoost 集成模型"]
    vals = [acc_const, acc_model]
    bars = ax.barh(names, vals, color=[ORANGE, BLUE], height=0.45, edgecolor="white")
    for r, v in zip(bars, vals):
        ax.text(v + 0.4, r.get_y() + r.get_height() / 2, f"{v:.2f}%", va="center", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("测试集准确率 / %")
    ax.set_title("（b）模型准确率低于常数基线")
    ax.grid(axis="x", alpha=0.35)
    ax.annotate("模型准确率反而更低", xy=(acc_model, 0.86), xytext=(26, 0.5),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig4_imbalance_baseline.jpg"))


# ---------------------------------------------------------------------------
# 二、决策树
# ---------------------------------------------------------------------------
def fig_tree_principle():
    rs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # (a) 特征空间的递归划分
    ax = axes[0]
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal((0.3, 0.75), 0.16, (40, 2)),
                   rng.normal((0.72, 0.3), 0.16, (40, 2)),
                   rng.normal((0.8, 0.82), 0.14, (30, 2))])
    y = np.r_[np.zeros(40, int), np.zeros(40, int), np.ones(30, int)]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=20, c=ORANGE, edgecolors="white", linewidths=0.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=20, c=BLUE, edgecolors="white", linewidths=0.3)
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, edgecolor="#333333", lw=1.4))
    ax.plot([0.55, 0.55], [0, 1], color="#333333", lw=1.4)
    ax.plot([0.55, 1.0], [0.55, 0.55], color="#333333", lw=1.4)
    ax.plot([0, 0.55], [0.55, 0.55], color="#333333", lw=1.4)
    bb = dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.78)
    ax.text(0.03, 0.97, "区域 1：判为 0", ha="left", va="top", fontsize=9, color=ORANGE, bbox=bb)
    ax.text(0.97, 0.03, "区域 2：判为 0", ha="right", va="bottom", fontsize=9, color=ORANGE, bbox=bb)
    ax.text(0.97, 0.97, "区域 3：判为 1", ha="right", va="top", fontsize=9, color=BLUE, bbox=bb)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("（a）决策树 = 特征空间的递归矩形划分")
    ax.grid(alpha=0.3)

    # (b) 熵与基尼指数
    ax = axes[1]
    p = np.linspace(0.001, 0.999, 400)
    ax.plot(p, -(p * np.log2(p) + (1 - p) * np.log2(1 - p)), color=BLUE, lw=2,
            label=r"熵 $Ent=-p\log_2p-(1-p)\log_2(1-p)$")
    ax.plot(p, 1 - (p ** 2 + (1 - p) ** 2), color=ORANGE, lw=2, ls="--",
            label=r"基尼指数 $Gini=1-p^2-(1-p)^2$")
    ax.axvline(0.5, color="#999999", ls=":", lw=1)
    ax.annotate("两类各占一半时不确定性最大", xy=(0.5, 1.0), xytext=(0.06, 0.86),
                fontsize=9.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    ax.set_xlabel("正类比例 $p$")
    ax.set_ylabel("不纯度")
    ax.set_title("（b）节点不纯度的两种度量")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8.5, loc="lower center")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig4_tree_principle.jpg"))


def fig_florence_overview():
    rs.apply()
    df = load_florence()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))
    ax = axes[0]
    vc = df["Florence"].value_counts().sort_index()
    bars = ax.bar(["未购买（0）", "购买（1）"], vc.values, color=[ORANGE, BLUE],
                  width=0.5, edgecolor="white")
    for r, v in zip(bars, vc.values):
        ax.text(r.get_x() + r.get_width() / 2, v + 60, f"{v}\n({v / len(df) * 100:.2f}%)",
                ha="center", fontsize=10)
    ax.set_ylim(0, 4300)
    ax.set_ylabel("样本数")
    ax.set_title("（a）是否购买《Florence》的分布")
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.78, 4250, "购买样本仅占 8.45%，正负样本比约 1 : 10.8", ha="center", va="top",
            fontsize=9.5, color="#333333", bbox=dict(boxstyle="round,pad=0.35",
                                                     facecolor="#f7f7f7", edgecolor="#c9c9c9", lw=0.8))

    ax = axes[1]
    feats = ["M", "R", "F"]
    data0 = [df.loc[df["Florence"] == 0, f].values for f in feats]
    data1 = [df.loc[df["Florence"] == 1, f].values for f in feats]
    pos = np.arange(len(feats))
    bp0 = ax.boxplot(data0, positions=pos - 0.18, widths=0.3, patch_artist=True,
                     showfliers=False, medianprops=dict(color=RED, lw=1.3))
    bp1 = ax.boxplot(data1, positions=pos + 0.18, widths=0.3, patch_artist=True,
                     showfliers=False, medianprops=dict(color=RED, lw=1.3))
    for b in bp0["boxes"]:
        b.set_facecolor(ORANGE)
        b.set_edgecolor("#b06a35")
    for b in bp1["boxes"]:
        b.set_facecolor(BLUE)
        b.set_edgecolor("#4a7fb5")
    ax.set_xticks(pos, ["M（金额）", "R（最近购买）", "F（频次）"])
    ax.set_ylabel("特征取值")
    ax.set_title("（b）三个 RFM 特征在两类样本上的分布")
    ax.legend([bp0["boxes"][0], bp1["boxes"][0]], ["未购买", "购买"], fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig4_florence_overview.jpg"))


def fig_tree_viz():
    """全特征决策树的前三层（自绘紧凑版）。"""
    rs.apply()
    df = load_florence()
    Xtr, Xte, ytr, yte = split_florence(df)
    est = make_tree().fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    draw_compact_tree(ax, est, list(Xtr.columns), ["未购买", "购买"], max_depth=3)
    ax.set_title("决策树（全部特征）的前三层结构", fontsize=11)
    rs.save(fig, os.path.join(OUT, "fig4_tree_viz.jpg"))


def fig_tree_3feat():
    """随机 3 个特征决策树的前三层（自绘紧凑版）。"""
    rs.apply()
    df = load_florence()
    random.seed(RS)
    selected = random.sample(df.drop(columns=["Seq#", "Florence"]).columns.tolist(), 3)
    Xtr, Xte, ytr, yte = split_florence(df, selected)
    est = make_tree().fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    draw_compact_tree(ax, est, selected, ["未购买", "购买"], max_depth=3)
    ax.set_title(f"随机选取 3 个特征（{', '.join(selected)}）的决策树前三层", fontsize=11)
    rs.save(fig, os.path.join(OUT, "fig4_tree_3feat.jpg"))
    return selected


def fig_importance():
    rs.apply()
    df = load_florence()
    Xtr, Xte, ytr, yte = split_florence(df)
    est = make_tree().fit(Xtr, ytr)
    imp = pd.Series(est.feature_importances_, index=Xtr.columns).sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.barh(imp.index, imp.values, color=[BLUE if v == imp.max() else "#9dc0e0" for v in imp.values],
            height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("特征重要性")
    ax.set_title("决策树各特征的重要性（全特征模型）")
    ax.set_xlim(0, max(imp.values) * 1.22)
    ax.grid(axis="x", alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig4_importance.jpg"))


# ===========================================================================
# 公式
# ===========================================================================
FORMULAS = {
    # ---- AdaBoost ----
    "f4_ensemble": r"$H(\mathbf{x})=\mathrm{sign}\left(\sum_{t=1}^{T}\alpha_th_t(\mathbf{x})\right)$",
    "f4_eps": r"$\varepsilon_t=\sum_{i=1}^{m}w_i^{(t)}\,\mathrm{I}\left(h_t(\mathbf{x}_i)\neq y_i\right)$",
    "f4_alpha": r"$\alpha_t=\dfrac{1}{2}\ln\dfrac{1-\varepsilon_t}{\varepsilon_t}$",
    "f4_wupdate": r"$w_i^{(t+1)}=\dfrac{w_i^{(t)}\exp\left(-\alpha_ty_ih_t(\mathbf{x}_i)\right)}{Z_t}$",
    "f4_Z": r"$Z_t=\sum_{i=1}^{m}w_i^{(t)}\exp\left(-\alpha_ty_ih_t(\mathbf{x}_i)\right)$",
    "f4_stump": r"$h_t(\mathbf{x})\;:\;\mathrm{max\_depth}=1\quad(\mathrm{stump})$",
    "f4_impute": r"$x_{\mathrm{miss}}\leftarrow\mathrm{median}(\mathbf{x}),\qquad x_{\mathrm{miss}}\leftarrow\mathrm{mode}(\mathbf{x})$",
    "f4_onehot": r"$\mathcal{X}_{\mathrm{cat}}\;\Longrightarrow\;\{0,1\}^{V}\quad(\mathrm{OneHot})$",
    # ---- 评估指标 ----
    "f4_acc": r"$acc=\dfrac{TP+TN}{TP+TN+FP+FN}$",
    "f4_prf": r"$P=\dfrac{TP}{TP+FP},\qquad R=\dfrac{TP}{TP+FN},\qquad F1=\dfrac{2PR}{P+R}$",
    "f4_macro": r"$F1_{\mathrm{macro}}=\dfrac{1}{K}\sum_{k=1}^{K}F1_k,\qquad F1_{\mathrm{weighted}}=\sum_{k=1}^{K}\dfrac{n_k}{N}F1_k$",
    "f4_baseline": r"$acc_{\mathrm{const}}=\max_k\dfrac{n_k}{N}$",
    "f4_ratio": r"$\mathrm{IR}=\dfrac{n_{-}}{n_{+}}$",
    # ---- 决策树 ----
    "f4_entropy": r"$Ent(D)=-\sum_{k=1}^{\left|\mathcal{Y}\right|}p_k\log_2p_k$",
    "f4_infogain": r"$Gain(D,a)=Ent(D)-\sum_{v=1}^{V}\dfrac{\left|D^v\right|}{\left|D\right|}Ent\left(D^v\right)$",
    "f4_gini": r"$Gini(D)=1-\sum_{k=1}^{\left|\mathcal{Y}\right|}p_k^2$",
    "f4_classweight": r"$w_k=\dfrac{N}{K\cdot N_k}$",
    "f4_leaf": r"$\hat{y}=\arg\max_{k}\;p\left(k\,|\,\text{叶节点}\right)$",
    "f4_prune": r"$\mathrm{max\_depth},\;\mathrm{min\_samples\_split},\;\mathrm{min\_samples\_leaf}\;\Longrightarrow\;\mathrm{complexity}$",
}


def render_formulas():
    print("渲染公式 ...")
    rs.apply()
    out_dir = os.path.join(rs.ASSETS, "exp4", "formula")
    os.makedirs(out_dir, exist_ok=True)
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=20, color="black")
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300,
                    bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
    print(f"  共 {len(FORMULAS)} 个公式 -> 素材/exp4/formula/")


if __name__ == "__main__":
    print("绘制实验四理论原理图 ...")
    fig_adaboost_principle()
    fig_pipeline()
    fig_imbalance_baseline()
    fig_tree_principle()
    fig_florence_overview()
    fig_tree_viz()
    fig_tree_3feat()
    fig_importance()
    render_formulas()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
