# -*- coding: utf-8 -*-
"""实验二素材生成（二）：自绘结果图。

notebook 只为 SVM 画了混淆矩阵图（cell 18），感知机的混淆矩阵仅有文本输出，
这里按同一视觉规范补一张热力图，并标注准确率、查准率、查全率与 F1。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import report_style as rs

OUT = os.path.join(rs.ASSETS, "exp2", "concept")
os.makedirs(OUT, exist_ok=True)


def fig_perceptron_cm():
    rs.apply()
    cm = np.array([[80, 20], [22, 32]])          # notebook cell 30 的实测混淆矩阵
    labels = ["未患糖尿病", "患糖尿病"]
    acc = np.trace(cm) / cm.sum() * 100
    prec = cm[1, 1] / cm[:, 1].sum()
    rec = cm[1, 1] / cm[1, :].sum()
    f1 = 2 * prec * rec / (prec + rec)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    ax.set_xticks([0, 1], [f"预测为\n{l}" for l in labels])
    ax.set_yticks([0, 1], [f"真实为\n{l}" for l in labels])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({v / cm.sum() * 100:.1f}%)", ha="center", va="center",
                    fontsize=12, color="white" if v > cm.max() * 0.6 else "#1a1a1a")
    ax.set_title(f"感知机在糖尿病测试集上的混淆矩阵\n"
                 f"准确率 {acc:.2f}%　查准率 {prec:.2f}　查全率 {rec:.2f}　F1 {f1:.2f}",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, label="样本数")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_perceptron_cm.jpg"))


if __name__ == "__main__":
    print("生成实验二自绘结果图 ...")
    fig_perceptron_cm()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
