# -*- coding: utf-8 -*-
"""改进 v2：全部改为自绘图。

1) 替换 4 张课件截图：混淆矩阵、P-R 曲线、ROC/AUC、10 折交叉验证（另补分层留出法）
2) 新增"散点 + 决策边界 + 误分类样本"结果图，替换 notebook 中那张只有一条直线的输出
所有图使用 report_style 统一主题，600 dpi，白底。
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

import report_style as rs

OUT = os.path.join(rs.ASSETS, "concept")
os.makedirs(OUT, exist_ok=True)
CODE_FONT = "DejaVu Sans"

# 复用 notebook 的划分方式，保证图与报告中的数值同源
RS_SPLIT, RS_LDA = 10086, 42


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------
def oof_proba(X, y, n_splits=5, seed=RS_SPLIT):
    """分层 K 折交叉验证的外推概率，用于画出较平滑的 P-R / ROC 曲线。"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        p[te] = LogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def rbox(ax, x, y, w, h, text, fc, fs=10, color="#1a1a1a", weight="normal", rot=0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#8a8a8a", lw=0.9))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=color, fontweight=weight, rotation=rot)


# ---------------------------------------------------------------------------
# 1. 混淆矩阵（替换课件截图）
# ---------------------------------------------------------------------------
def fig_confusion_matrix():
    rs.apply()
    fig = plt.figure(figsize=(9.8, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.03)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.set_axis_off()
    head, hi, lo = "#eef3f9", "#dce9f6", "#f7e2d5"
    X_LAB, W_LAB = 0.15, 2.3
    X_C1, X_C2, W_C = 2.45, 6.2, 3.75
    H_H1, H_H2, H_R = 0.95, 0.9, 1.1
    Y_H1, Y_H2 = 5.55, 4.65
    Y_R1, Y_R2 = 3.55, 2.45

    ax.text(5.0, 7.05, "（a）二分类问题的混淆矩阵", ha="center", va="center", fontsize=11.5)
    rbox(ax, X_LAB, Y_H2, W_LAB, H_H1 + H_H2, "真实情况", head, 10.5, rot=90)
    rbox(ax, X_C1, Y_H1, 2 * W_C, H_H1, "预测结果", head, 10.5)
    rbox(ax, X_C1, Y_H2, W_C, H_H2, "正例", head, 10)
    rbox(ax, X_C2, Y_H2, W_C, H_H2, "反例", head, 10)
    rbox(ax, X_LAB, Y_R1, W_LAB, H_R, "正例", "#f7f7f7", 10)
    rbox(ax, X_LAB, Y_R2, W_LAB, H_R, "反例", "#f7f7f7", 10)
    rbox(ax, X_C1, Y_R1, W_C, H_R, "TP（真正例）", hi, 10)
    rbox(ax, X_C2, Y_R1, W_C, H_R, "FN（假反例）", lo, 10)
    rbox(ax, X_C1, Y_R2, W_C, H_R, "FP（假正例）", lo, 10)
    rbox(ax, X_C2, Y_R2, W_C, H_R, "TN（真反例）", hi, 10)
    ax.text(0.15, 1.75, "TP 真正例：正例被判为正例；　FN 假反例：正例被漏判", fontsize=9.5, color="#333333")
    ax.text(0.15, 1.05, "FP 假正例：反例被误判为正例；TN 真反例：反例被判为反例", fontsize=9.5, color="#333333")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 7.4)
    ax2.set_axis_off()
    ax2.text(5.0, 7.05, "（b）由混淆矩阵导出的指标", ha="center", va="center", fontsize=11.5)
    for yy, name, tex in (
        (5.5, "查准率（Precision）", r"$P=\dfrac{TP}{TP+FP}$"),
        (4.1, "查全率（Recall）", r"$R=\dfrac{TP}{TP+FN}$"),
        (2.7, "F1 分数", r"$F1=\dfrac{2\times P\times R}{P+R}$"),
    ):
        ax2.text(0.3, yy, name, ha="left", va="center", fontsize=10)
        ax2.text(7.0, yy, tex, ha="center", va="center", fontsize=17)
    ax2.text(0.4, 1.2, "本实验 LDA 实测：混淆矩阵为对角阵（对角元素均为 10），\n"
                       "故各类查准率、查全率、F1 均为 1.00",
             ha="left", va="center", fontsize=9.5, color="#333333")
    rs.save(fig, os.path.join(OUT, "fig_confusion_matrix.jpg"))


# ---------------------------------------------------------------------------
# 2. P-R 曲线（替换课件截图）
# ---------------------------------------------------------------------------
def fig_pr_curve():
    rs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # (a) 原理示意：三条 P-R 曲线与平衡点
    ax = axes[0]
    r = np.linspace(0.01, 1.0, 400)
    curves = {
        "A": (1 - 0.35 * r ** 1.6, rs.BLUE),
        "B": (1 - 0.55 * r ** 0.35, rs.ORANGE),
        "C": (1 - 0.95 * r ** 1.6, rs.GREEN),
    }
    ax.plot(r, r, ls="--", lw=1.1, color="#b0b0b0", label="随机猜测（P = R 对角线）")
    for no, (name, (p, c)) in enumerate(curves.items()):
        ax.plot(r, p, color=c, lw=2, label=f"曲线 {name}")
        idx = int(np.argmin(np.abs(p - r)))
        ax.plot(r[idx], p[idx], "o", ms=5.5, color=c, markeredgecolor="white",
                markeredgewidth=0.8, label="平衡点（P = R）" if no == 0 else None)
        ax.plot([r[idx], r[idx]], [0, p[idx]], ls=":", lw=0.9, color=c)
        ax.plot([0, r[idx]], [p[idx], p[idx]], ls=":", lw=0.9, color=c)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("查全率（Recall）")
    ax.set_ylabel("查准率（Precision）")
    ax.set_title("（a）P-R 曲线与平衡点（原理示意）")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.5)

    # (b) 本实验数据实测
    ax = axes[1]
    X, y = rs.simmons()
    proba = oof_proba(X, y)
    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    ax.plot(rec, prec, color=rs.BLUE, lw=2, label=f"逻辑回归（AP = {ap:.3f}）")
    ax.axhline(y.mean(), ls="--", lw=1.1, color="#b0b0b0",
               label=f"随机基线（正例比例 = {y.mean():.2f}）")
    k = int(np.argmin(np.abs(prec[:-1] - rec[:-1])))
    ax.plot(rec[k], prec[k], "o", ms=6, color=rs.RED, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(f"平衡点 P = R = {prec[k]:.2f}", xy=(rec[k], prec[k]),
                xytext=(rec[k] - 0.42, prec[k] + 0.16), fontsize=9, color=rs.RED,
                arrowprops=dict(arrowstyle="->", color=rs.RED, lw=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("查全率（Recall）")
    ax.set_ylabel("查准率（Precision）")
    ax.set_title("（b）本实验数据实测（5 折交叉验证外推概率）")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.5)

    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig_pr_curve.jpg"))


# ---------------------------------------------------------------------------
# 3. ROC 曲线与 AUC（替换课件截图）
# ---------------------------------------------------------------------------
def fig_roc_auc():
    rs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    X, y = rs.simmons()

    # (a) 实测 ROC + AUC 面积 + 梯形积分公式
    ax = axes[0]
    proba = oof_proba(X, y)
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    ax.fill_between(fpr, tpr, color=rs.BLUE, alpha=0.16)
    ax.plot(fpr, tpr, color=rs.BLUE, lw=2.2, label=f"Spending + Card（AUC = {auc:.3f}）")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#b0b0b0", label="随机猜测（AUC = 0.5）")
    ax.text(0.30, 0.10, r"$AUC=\dfrac{1}{2}\sum_{i=1}^{m-1}(x_{i+1}-x_i)(y_i+y_{i+1})$",
            fontsize=11, color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#c9c9c9"))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("假正例率（FPR）")
    ax.set_ylabel("真正例率（TPR）")
    ax.set_title("（a）ROC 曲线与 AUC（本实验数据实测）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.5)

    # (b) 特征选择对排序能力的影响
    ax = axes[1]
    proba1 = oof_proba(X[:, :1], y)
    fpr1, tpr1, _ = roc_curve(y, proba1)
    auc1 = roc_auc_score(y, proba1)
    ax.plot(fpr, tpr, color=rs.BLUE, lw=2.2, label=f"Spending + Card（AUC = {auc:.3f}）")
    ax.plot(fpr1, tpr1, color=rs.ORANGE, lw=2.2, ls="-.", label=f"仅 Spending（AUC = {auc1:.3f}）")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#b0b0b0")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("假正例率（FPR）")
    ax.set_ylabel("真正例率（TPR）")
    ax.set_title("（b）特征选择对排序能力的影响")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.5)

    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig_roc_auc.jpg"))


# ---------------------------------------------------------------------------
# 4. 评估方法：10 折交叉验证 + 分层留出法（替换课件截图）
# ---------------------------------------------------------------------------
def fig_cross_validation():
    rs.apply()
    fig = plt.figure(figsize=(9.6, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.30)

    # (a) 10 折交叉验证
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.text(0, 7.7, "（a）10 折交叉验证：轮流留出一折作测试集，最终取 k 个测试结果的平均",
            fontsize=11, va="top")
    n, x0, w, h = 10, 0.9, 0.82, 0.68
    rbox(ax, 0.9, 6.45, n * w, 0.7, "数据集 $D$（$m$ 个样本）", "#eef3f9", 10)
    ax.annotate("", xy=(0.9 + n * w / 2, 6.1), xytext=(0.9 + n * w / 2, 6.45),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2))
    for i in range(n):
        rbox(ax, x0 + i * w, 5.35, w, h, f"$D_{{{i+1}}}$", "#dce9f6", 8.5)
    for k, label, yy in ((1, "第 1 折", 3.95), (2, "第 2 折", 2.85), (10, "第 10 折", 1.75)):
        ax.text(0.82, yy + h / 2, label, ha="right", va="center", fontsize=9.5)
        for i in range(n):
            test = (i == k - 1)
            rbox(ax, x0 + i * w, yy, w, h, "", "#f6d9d2" if test else "#f2f2f2")
            ax.add_patch(Rectangle((x0 + i * w, yy), w, h, facecolor="none",
                                   edgecolor=rs.RED if test else "#b8b8b8",
                                   lw=1.3 if test else 0.7))
        ax.annotate("", xy=(9.9, yy + h / 2), xytext=(9.45, yy + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0))
        ax.text(10.05, yy + h / 2, f"测试结果 {k}", ha="left", va="center", fontsize=9.5)
    ax.text(x0 + n * w / 2, 2.64, r"$\vdots$", ha="center", va="center", fontsize=13)
    ax.plot([12.15, 12.15], [1.75 + h / 2, 3.95 + h / 2], color="#444444", lw=1.1)
    ax.plot([12.15, 12.45], [1.75 + h / 2, 1.75 + h / 2], color="#444444", lw=1.1)
    ax.plot([12.15, 12.45], [3.95 + h / 2, 3.95 + h / 2], color="#444444", lw=1.1)
    ax.annotate("", xy=(13.5, 2.85 + h / 2), xytext=(12.5, 2.85 + h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.1))
    ax.text(12.6, 3.32, "平均", fontsize=9.5)
    ax.text(13.55, 2.85 + h / 2, "返回结果", ha="left", va="center", fontsize=9.5)
    ax.text(0.9, 0.75, "红色为留出的测试集；每次用其余 9 折作为训练集。",
            fontsize=9.5, color="#333333")

    # (b) 分层留出法（对应本实验代码）
    ax2 = fig.add_subplot(gs[1])
    labels = ["原始数据集\n100 个样本", "训练集\n80 个样本", "测试集\n20 个样本"]
    n0, n1 = [50, 40, 10], [50, 40, 10]
    ypos = [2, 1, 0]
    ax2.barh(ypos, n0, color=rs.ORANGE, height=0.55, edgecolor="white")
    ax2.barh(ypos, n1, left=n0, color=rs.BLUE, height=0.55, edgecolor="white")
    for yy, a, b in zip(ypos, n0, n1):
        ax2.text(a / 2, yy, f"{a}", ha="center", va="center", color="white", fontsize=10)
        ax2.text(a + b / 2, yy, f"{b}", ha="center", va="center", color="white", fontsize=10)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=9.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.6, 2.75)
    ax2.set_xlabel("样本占比 / %")
    ax2.set_title("（b）分层留出法：按 8 : 2 划分且保持类别比例"
                  "（对应 train_test_split(test_size=0.2, stratify=y)）", fontsize=10.5)
    ax2.legend(handles=[Patch(color=rs.ORANGE, label="Coupon = 0（不使用优惠券）"),
                        Patch(color=rs.BLUE, label="Coupon = 1（使用优惠券）")],
               loc="lower right", ncol=1, fontsize=9)
    ax2.grid(axis="x", alpha=0.5)
    rs.save(fig, os.path.join(OUT, "fig_cross_validation.jpg"))


# ---------------------------------------------------------------------------
# 5. 决策边界的运行结果：散点 + 边界 + 误分类样本（替换 notebook 中的直线图）
# ---------------------------------------------------------------------------
def fig_boundary_result():
    rs.apply()
    X, y = rs.simmons()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=RS_SPLIT, stratify=y)
    clf = LogisticRegression().fit(Xtr, ytr)
    yp = clf.predict(Xte)
    err = yp != yte
    w, b = clf.coef_[0], clf.intercept_[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=24, c=rs.ORANGE, alpha=0.35,
               edgecolors="none", label="训练集 Coupon = 0")
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=24, c=rs.BLUE, alpha=0.35,
               edgecolors="none", label="训练集 Coupon = 1")
    xs = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 200)
    ax.plot(xs, -(b + w[0] * xs) / w[1], color=rs.RED, lw=2, label="决策边界 $w^{T}x+b=0$")
    ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=52, c=rs.ORANGE,
               edgecolors="white", linewidths=0.8, label="测试集 Coupon = 0")
    ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=52, c=rs.BLUE,
               edgecolors="white", linewidths=0.8, label="测试集 Coupon = 1")
    ax.scatter(Xte[err, 0], Xte[err, 1], s=185, facecolors="none",
               edgecolors=rs.RED, linewidths=1.8, label=f"误分类样本（{int(err.sum())} 个）")
    ax.text(0.02, 0.04,
            f"测试集 {len(yte)} 个样本，错分 {int(err.sum())} 个，准确率 {(1 - err.mean()) * 100:.2f}%",
            transform=ax.transAxes, fontsize=10, color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf3ee", edgecolor=rs.RED, lw=0.8))
    ax.set_xlabel("Spending（消费额）")
    ax.set_ylabel("Card（持卡情况）")
    ax.set_title("决策边界的运行结果：测试集误分类样本")
    ax.set_xlim(X[:, 0].min() - 0.4, X[:, 0].max() + 0.5)
    ax.set_ylim(-0.65, 1.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=8.5, frameon=False)
    ax.grid(alpha=0.4)
    rs.save(fig, os.path.join(OUT, "fig_boundary_result.jpg"))


if __name__ == "__main__":
    print("生成自绘概念图与结果图 ...")
    fig_confusion_matrix()
    fig_pr_curve()
    fig_roc_auc()
    fig_cross_validation()
    fig_boundary_result()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
