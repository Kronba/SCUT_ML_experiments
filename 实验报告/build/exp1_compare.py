# -*- coding: utf-8 -*-
"""实验一：调包实现 vs 自实现 的对比图（600 dpi，统一 report_style 主题）。

所有数值都与 exp1.ipynb 的执行结果同源（相同的划分、随机种子与参数）。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
from sklearn.linear_model import LogisticRegression as SkLogit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import report_style as rs
from models.exp1.LDA import LinearDiscriminantAnalysis as MyLDA
from models.exp1.sigmoid import LogisticRegression as MyLogit, sigmoid as my_sigmoid

OUT = os.path.join(rs.ASSETS, "exp1", "compare")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED


def simmons():
    d = pd.read_excel(os.path.join(rs.PROJ_DIR, "data", "exp1", "Simmons.xlsx"))
    X, y = d[["Spending", "Card"]].values, d["Coupon"].values
    return d, X, y


def fit_logit():
    d, X, y = simmons()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=10086, stratify=y)
    sk = SkLogit().fit(Xtr, ytr)
    newton = MyLogit(solver="newton", C=1.0, n_iters=200, tol=1e-8).fit(Xtr, ytr)
    gd = MyLogit(solver="gd", learning_rate=0.005, n_iters=5000, tol=1e-8).fit(Xtr, ytr)
    return d, Xtr, Xte, ytr, yte, sk, newton, gd


def fit_lda():
    iris = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2,
                                          random_state=42, stratify=iris.target)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    sk = SkLDA().fit(Xtr_s, ytr)
    mine = MyLDA().fit(Xtr_s, ytr)
    return iris, Xtr_s, Xte_s, ytr, yte, sk, mine


# ---------------------------------------------------------------------------
def fig_sigmoid_compare():
    rs.apply()
    z = np.linspace(-10, 10, 200)
    y_formula = 1 / (1 + np.exp(-z))
    y_mine = my_sigmoid(z)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.plot(z, y_formula, color=BLUE, lw=4, alpha=0.55, label="公式实现")
    ax.plot(z, y_mine, "--", color=RED, lw=1.8, label="models/exp1/sigmoid.py")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\\sigma(z)$")
    ax.set_title("（a）两种实现完全重合")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.35)
    ax = axes[1]
    ax.semilogy(z, np.abs(y_formula - y_mine) + 1e-18, color=GREEN, lw=1.8)
    ax.set_xlabel("$z$")
    ax.set_ylabel("绝对误差（对数坐标）")
    ax.set_title(r"（b）最大绝对误差 $5.55\times10^{-17}$")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig1_sigmoid_compare.jpg"))


def fig_convergence():
    rs.apply()
    *_, sk, newton, gd = fit_logit()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(newton.loss_history_, color=RED, lw=2, marker="o", markersize=4,
            label=f"牛顿法（{newton.n_iter_} 轮）")
    ax.plot(gd.loss_history_, color=BLUE, lw=1.8,
            label=f"梯度下降（{gd.n_iter_} 轮）")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("目标函数值")
    ax.set_xlim(-30, 1520)
    ax.set_title("自实现的收敛过程")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig1_convergence.jpg"))


def fig_boundary_compare():
    rs.apply()
    d, Xtr, Xte, ytr, yte, sk, newton, gd = fit_logit()
    xs = np.linspace(d["Spending"].min() - 0.2, d["Spending"].max() + 0.2, 200)
    b_sk = -(sk.intercept_[0] + sk.coef_[0][0] * xs) / sk.coef_[0][1]
    b_my = -(newton.intercept_[0] + newton.coef_[0][0] * xs) / newton.coef_[0][1]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax = axes[0]
    ax.scatter(d.loc[d["Coupon"] == 0, "Spending"], d.loc[d["Coupon"] == 0, "Card"],
               s=26, c=ORANGE, edgecolors="white", linewidths=0.5, label="Coupon = 0")
    ax.scatter(d.loc[d["Coupon"] == 1, "Spending"], d.loc[d["Coupon"] == 1, "Card"],
               s=26, c=BLUE, edgecolors="white", linewidths=0.5, label="Coupon = 1")
    ax.plot(xs, b_sk, "k--", lw=3.4, label="sklearn 调包")
    ax.plot(xs, b_my, "-", color=RED, lw=1.4, label="自实现（牛顿法）")
    ax.set_xlabel("Spending")
    ax.set_ylabel("Card")
    ax.set_title(f"（a）决策边界对比（最大差异 {np.max(np.abs(b_sk - b_my)):.2e}）")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.35)

    ax = axes[1]
    p_sk = sk.predict_proba(Xte)[:, 1]
    p_my = newton.predict_proba(Xte)[:, 1]
    ax.scatter(p_sk, p_my, s=40, c=BLUE, edgecolors="white", linewidths=0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("sklearn 预测的正类概率")
    ax.set_ylabel("自实现预测的正类概率")
    ax.set_title(f"（b）测试集概率对比（最大差异 {np.max(np.abs(p_sk - p_my)):.2e}）")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig1_boundary_compare.jpg"))


def fig_lda_cm_compare():
    rs.apply()
    iris, Xtr_s, Xte_s, ytr, yte, sk, mine = fit_lda()
    cms = [(confusion_matrix(yte, sk.predict(Xte_s)), "sklearn 调包"),
           (confusion_matrix(yte, mine.predict(Xte_s)), "自实现")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, (cm, title) in zip(axes, cms):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "#1a1a1a")
        ax.set_xticks(range(3), iris.target_names, fontsize=8.5)
        ax.set_yticks(range(3), iris.target_names, fontsize=8.5)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"（{'a' if title.startswith('sklearn') else 'b'}）{title}")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig1_lda_cm_compare.jpg"))


def fig_lda_proj_compare():
    rs.apply()
    iris, Xtr_s, Xte_s, ytr, yte, sk, mine = fit_lda()
    Z_sk = sk.transform(Xtr_s)
    Z_my = mine.transform(Xtr_s)
    Z_my = Z_my * np.sign(np.sum(Z_sk * Z_my, axis=0))       # 统一符号
    Z_my = Z_my / Z_my.std(axis=0) * Z_sk.std(axis=0)        # 统一尺度后比较

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (Z, title) in zip(axes, [(Z_sk, "sklearn 调包"), (Z_my, "自实现")]):
        for k, name in enumerate(iris.target_names):
            ax.scatter(Z[ytr == k, 0], Z[ytr == k, 1], s=26, label=name,
                       edgecolors="white", linewidths=0.4)
        ax.set_xlabel("Linear Discriminant 1")
        ax.set_ylabel("Linear Discriminant 2")
        ax.set_title(f"（{'a' if title.startswith('sklearn') else 'b'}）LDA 投影：{title}")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig1_lda_proj_compare.jpg"))


if __name__ == "__main__":
    print("生成实验一对比图 ...")
    fig_sigmoid_compare()
    fig_convergence()
    fig_boundary_compare()
    fig_lda_cm_compare()
    fig_lda_proj_compare()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
