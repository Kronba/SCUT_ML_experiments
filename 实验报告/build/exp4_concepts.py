# -*- coding: utf-8 -*-
"""实验四素材（二）：结果类自绘图。

1) AdaBoost 各类别的查准率/查全率/F1（notebook 保存的实测结果）
2) 决策树是否启用 class_weight="balanced" 的对比（真实实验，用于说明“平衡因子”的作用）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import exp4_assets as A
import report_style as rs

OUT = os.path.join(rs.ASSETS, "exp4", "concept")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED


def fig_metrics_bars():
    """AdaBoost 在测试集上各类别的 P/R/F1（notebook cell 17 保存的结果）。"""
    rs.apply()
    metrics = ["查准率 P", "查全率 R", "F1 分数"]
    normal = [0.89, 1.00, 0.94]
    fraud = [0.00, 0.00, 0.00]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bars0 = ax.bar(x - 0.19, normal, width=0.38, color=ORANGE, label="Normal（正常交易）")
    bars1 = ax.bar(x + 0.19, fraud, width=0.38, color=BLUE, label="Fraud（欺诈交易）")
    for bars in (bars0, bars1):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                    f"{r.get_height():.2f}", ha="center", fontsize=9.5)
    for r in bars1:                      # 0 高度的柱子加一条可见的细条
        ax.add_patch(plt.Rectangle((r.get_x(), 0), r.get_width(), 0.012,
                                   facecolor=BLUE, edgecolor="none"))
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.38)
    ax.set_ylabel("指标取值")
    ax.set_title("AdaBoost 各类别的评估指标（宏平均 F1 仅 0.47）")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.5, 1.35, "欺诈类三项指标全为 0：53 个欺诈样本无一被识别",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=RED,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdf3ee", edgecolor=RED, lw=0.8))
    rs.save(fig, os.path.join(OUT, "fig4_metrics_bars.jpg"))


def fig_balanced_compare():
    """class_weight 平衡前后，少数类（购买）识别能力的变化（真实实验）。"""
    rs.apply()
    df = A.load_florence()
    Xtr, Xte, ytr, yte = A.split_florence(df)

    results = {}
    for tag, cw in (("不加平衡因子", None), ("class_weight=\"balanced\"", "balanced")):
        est = A.DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=10,
                                       min_samples_leaf=5, class_weight=cw,
                                       random_state=A.RS).fit(Xtr, ytr)
        pred = est.predict(Xte)
        rep = classification_report(yte, pred, output_dict=True)
        results[tag] = {
            "acc": rep["accuracy"] * 100,
            "P": rep["1"]["precision"],
            "R": rep["1"]["recall"],
            "F1": rep["1"]["f1-score"],
            "n_pred_pos": int(pred.sum()),
        }
        print(f"    [信息] {tag}: 准确率 {rep['accuracy']*100:.2f}%，"
              f"购买类 P={rep['1']['precision']:.2f} R={rep['1']['recall']:.2f} "
              f"F1={rep['1']['f1-score']:.2f}，预测为购买的样本 {int(pred.sum())} 个")

    tags = list(results.keys())
    metrics = ["P", "R", "F1"]
    metric_labels = ["查准率 P", "查全率 R", "F1 分数"]
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))
    ax = axes[0]
    ax.bar(x - 0.19, [results[tags[0]][m] for m in metrics], width=0.38, color=ORANGE, label=tags[0])
    ax.bar(x + 0.19, [results[tags[1]][m] for m in metrics], width=0.38, color=BLUE, label="balanced 平衡权重")
    for i, m in enumerate(metrics):
        for dx, tag in ((-0.19, tags[0]), (0.19, tags[1])):
            v = results[tag][m]
            ax.text(i + dx, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("购买类（少数类）的指标")
    ax.set_title("（a）少数类的识别能力")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    accs = [results[t]["acc"] for t in tags]
    bars = ax.bar(["不加平衡因子", "balanced 平衡权重"], accs, color=[ORANGE, BLUE],
                  width=0.5, edgecolor="white")
    for r, v, t in zip(bars, accs, tags):
        ax.text(r.get_x() + r.get_width() / 2, v + 2.0,
                f"{v:.2f}%\n预测为购买 {results[t]['n_pred_pos']} / 68",
                ha="center", fontsize=9.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("测试集准确率 / %")
    ax.set_title("（b）总体准确率与少数类预测数量")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig4_balanced_compare.jpg"))


if __name__ == "__main__":
    print("生成实验四自绘结果图 ...")
    fig_metrics_bars()
    fig_balanced_compare()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
