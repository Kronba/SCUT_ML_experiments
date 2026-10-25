# -*- coding: utf-8 -*-
"""实验二：调包实现 vs 自实现 的对比图（600 dpi，统一 report_style 主题）。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron as SkPerc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SkSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import report_style as rs
from models.exp2.perceptron import Perceptron as MyPerc
from models.exp2.svm import SVC as MySVC

OUT = os.path.join(rs.ASSETS, "exp2", "compare")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED
RS = 42
AUTO_FEATS = ["cylinders", "displacement", "horsepower", "weight",
              "acceleration", "model_year", "origin"]


def load_auto():
    df = pd.read_csv(os.path.join(rs.PROJ_DIR, "data", "exp2", "auto-mpg.data"), sep=r"\s+",
                     names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                            "acceleration", "model_year", "origin", "car_name"], na_values="?")
    df = df.dropna()
    df["grade"] = (df["mpg"] >= df["mpg"].median()).astype(int)
    return df


def load_diabetes():
    return pd.read_csv(os.path.join(rs.PROJ_DIR, "data", "exp2", "diabetes.csv"))


def fig_svm_compare():
    rs.apply()
    # (a) 二维特征上的决策边界
    df = load_auto()
    feats = ["weight", "horsepower"]
    X = df[feats].values
    y = df["grade"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
    sc = StandardScaler()
    A, B = sc.fit_transform(Xtr), sc.transform(Xte)
    sk = SkSVC(kernel="rbf", C=1.0, gamma="scale").fit(A, ytr)
    mine = MySVC(kernel="rbf", C=1.0, gamma="scale", tol=1e-3, max_iter=5000).fit(A, ytr)

    lo, hi = A.min(0) - 0.4, A.max(0) + 0.4
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 350), np.linspace(lo[1], hi[1], 350))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Zs = sk.decision_function(grid).reshape(xx.shape)
    Zm = mine.decision_function(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax = axes[0]
    ax.contourf(xx, yy, Zs, levels=[-1e9, 0, 1e9], colors=["#fae3d0", "#d9e6f5"], alpha=0.55)
    ax.contour(xx, yy, Zs, levels=[0], colors=["#333333"], linewidths=2.6)
    ax.contour(xx, yy, Zm, levels=[0], colors=[RED], linewidths=1.4, linestyles="--")
    ax.scatter(A[ytr == 0, 0], A[ytr == 0, 1], s=16, c=ORANGE, alpha=0.8, edgecolors="none")
    ax.scatter(A[ytr == 1, 0], A[ytr == 1, 1], s=16, c=BLUE, alpha=0.8, edgecolors="none")
    ax.plot([], [], color="#333333", lw=2.6, label=f"sklearn（测试集 {sk.score(B, yte)*100:.2f}%）")
    ax.plot([], [], color=RED, lw=1.4, ls="--", label=f"自实现（测试集 {mine.score(B, yte)*100:.2f}%）")
    ax.set_xlabel("weight（标准化）")
    ax.set_ylabel("horsepower（标准化）")
    ax.set_title("（a）二维特征上的决策边界")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) 七特征模型：决策函数值对比
    X7 = df[AUTO_FEATS].values
    Xtr7, Xte7, ytr7, yte7 = train_test_split(X7, y, test_size=0.2, random_state=RS, stratify=y)
    sc7 = StandardScaler()
    A7, B7 = sc7.fit_transform(Xtr7), sc7.transform(Xte7)
    sk7 = SkSVC(kernel="rbf", C=1.0, gamma="scale").fit(A7, ytr7)
    my7 = MySVC(kernel="rbf", C=1.0, gamma="scale", tol=1e-3, max_iter=5000).fit(A7, ytr7)
    zs, zm = sk7.decision_function(B7), my7.decision_function(B7)
    ax = axes[1]
    ax.scatter(zs, zm, s=36, c=BLUE, edgecolors="white", linewidths=0.6)
    lim = [min(zs.min(), zm.min()) - 0.1, max(zs.max(), zm.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("sklearn 的决策函数值")
    ax.set_ylabel("自实现的决策函数值")
    ax.set_title(f"（b）七特征模型：决策值最大差异 {np.max(np.abs(zs - zm)):.2e}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_svm_compare.jpg"))


def fig_perceptron_compare():
    rs.apply()
    df = load_diabetes()
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
    sc = StandardScaler()
    A, B = sc.fit_transform(Xtr), sc.transform(Xte)
    sk = SkPerc(max_iter=1000, tol=1e-3, random_state=RS).fit(A, ytr)
    mine = MyPerc(max_iter=1000, tol=1e-3, random_state=RS).fit(A, ytr)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    # (a) 二维特征上的边界
    feats = ["Glucose", "BMI"]
    X2 = df[feats].values
    X2tr, X2te, y2tr, y2te = train_test_split(X2, y, test_size=0.2, random_state=RS, stratify=y)
    sc2 = StandardScaler()
    A2, B2 = sc2.fit_transform(X2tr), sc2.transform(X2te)
    p_sk = SkPerc(max_iter=1000, tol=1e-3, random_state=RS).fit(A2, y2tr)
    p_my = MyPerc(max_iter=1000, tol=1e-3, random_state=RS).fit(A2, y2tr)
    xs = np.linspace(A2[:, 0].min() - 0.3, A2[:, 0].max() + 0.3, 200)
    ax = axes[0]
    ax.scatter(A2[y2tr == 0, 0], A2[y2tr == 0, 1], s=14, c=ORANGE, alpha=0.55,
               edgecolors="none", label="未患病")
    ax.scatter(A2[y2tr == 1, 0], A2[y2tr == 1, 1], s=14, c=BLUE, alpha=0.55,
               edgecolors="none", label="患病")
    w, b = p_sk.coef_[0], p_sk.intercept_[0]
    ax.plot(xs, -(w[0] * xs + b) / w[1], color="#333333", lw=2.6, label="sklearn")
    w, b = p_my.coef_[0], p_my.intercept_[0]
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=RED, lw=1.5, ls="--", label="自实现")
    ax.set_xlabel("Glucose（标准化）")
    ax.set_ylabel("BMI（标准化）")
    ax.set_title("（a）二维特征上的决策边界")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) 不同随机种子下的准确率波动
    seeds = list(range(10)) + [RS]
    accs = []
    for seed in seeds:
        m = MyPerc(max_iter=1000, tol=1e-3, random_state=seed).fit(A, ytr)
        accs.append(accuracy_score(yte, m.predict(B)) * 100)
    ax = axes[1]
    labels = [str(s) for s in seeds]
    bars = ax.bar(labels, accs, color=BLUE, width=0.6, edgecolor="white")
    bars[labels.index(str(RS))].set_color(RED)
    ax.axhline(accuracy_score(yte, sk.predict(B)) * 100, color=GREEN, ls="--", lw=1.6,
               label=f"sklearn：{accuracy_score(yte, sk.predict(B)) * 100:.2f}%")
    ax.axhline(np.mean(accs), color="#666666", ls=":", lw=1.4,
               label=f"自实现均值：{np.mean(accs):.2f}%")
    ax.set_ylim(55, 80)
    ax.set_xlabel("随机种子")
    ax.set_ylabel("测试集准确率 / %")
    ax.set_title("（b）感知机解不唯一：不同更新顺序的准确率")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_perceptron_compare.jpg"))
    print(f"    [信息] 种子均值 {np.mean(accs):.2f}%，范围 {min(accs):.2f}%~{max(accs):.2f}%")


if __name__ == "__main__":
    print("生成实验二对比图 ...")
    fig_svm_compare()
    fig_perceptron_compare()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
