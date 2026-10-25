# -*- coding: utf-8 -*-
"""实验二素材生成（一）：SVM 与感知机的理论原理图 + 公式图。

统一使用 report_style 主题（白底、中文衬线、Computer Modern 数学字体、600 dpi）。
所有"实测类"插图都使用 data/exp2 下的真实数据，与 notebook 的划分/随机种子保持一致。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import report_style as rs

OUT = os.path.join(rs.ASSETS, "exp2", "theory")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED
RS = 42


# ---------------------------------------------------------------------------
# 数据与绘图公共工具
# ---------------------------------------------------------------------------
def load_auto():
    """Auto MPG 数据：读取、删除缺失、按 mpg 中位数构造 grade 二分类标签。"""
    df = pd.read_csv(
        os.path.join(rs.PROJ_DIR, "data", "exp2", "auto-mpg.data"), sep=r"\s+",
        names=["mpg", "cylinders", "displacement", "horsepower", "weight",
               "acceleration", "model_year", "origin", "car_name"], na_values="?")
    df = df.dropna()
    df["grade"] = (df["mpg"] >= df["mpg"].median()).astype(int)
    return df


def load_diabetes():
    return pd.read_csv(os.path.join(rs.PROJ_DIR, "data", "exp2", "diabetes.csv"))


FEATS2 = ["weight", "horsepower"]


def auto_2d():
    """取两个最具代表性的特征并标准化，用于二维决策边界的可视化。"""
    df = load_auto()
    X = df[FEATS2].values
    y = df["grade"].values
    Xs = StandardScaler().fit_transform(X)
    return Xs, y


def two_gaussians(n=60, sep=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-sep, -sep), scale=0.75, size=(n, 2))
    X1 = rng.normal(loc=(sep, sep), scale=0.75, size=(n, 2))
    return np.vstack([X0, X1]), np.r_[np.zeros(n, int), np.ones(n, int)]


def plot_boundary(ax, model, xlim, ylim, color=RED, lw=2, ls="-", levels=(0.0,),
                  label=None):
    """在给定坐标范围内画出决策函数等高线（0 为决策边界，±1 为间隔边界）。"""
    xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=list(levels), colors=[color],
               linewidths=lw, linestyles=ls)
    if label:                      # 用代理线做图例（contour 的 label 在新版 matplotlib 不可用）
        ax.plot([], [], color=color, lw=lw, ls=ls, label=label)
    return Z


def scatter2(ax, X, y, s=22, alpha=0.85, size_train=None):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=ORANGE, alpha=alpha,
               edgecolors="white", linewidths=0.4, label="grade = 0" if size_train is None else "类别 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, c=BLUE, alpha=alpha,
               edgecolors="white", linewidths=0.4, label="grade = 1" if size_train is None else "类别 1")


# ---------------------------------------------------------------------------
# 1. 最大间隔与支持向量
# ---------------------------------------------------------------------------
def fig_svm_margin():
    rs.apply()
    X, y = two_gaussians(n=22, sep=1.05, seed=3)
    yy_ = np.where(y == 0, -1, 1)
    clf = SVC(kernel="linear", C=1e4).fit(X, yy_)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    lo = X.min(0) - 0.8
    hi = X.max(0) + 0.8
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#fae3d0", "#d9e6f5"], alpha=0.55)
    plot_boundary(ax, clf, (lo[0], hi[0]), (lo[1], hi[1]), color="#333333", lw=1.8,
                  levels=[0.0], label=r"决策边界 $\mathbf{w}^{T}\mathbf{x}+b=0$")
    plot_boundary(ax, clf, (lo[0], hi[0]), (lo[1], hi[1]), color=RED, lw=1.3, ls="--",
                  levels=[-1.0, 1.0], label=r"间隔边界 $\pm 1$")
    scatter2(ax, X, (yy_ > 0).astype(int))
    sv = clf.support_
    ax.scatter(X[sv, 0], X[sv, 1], s=190, facecolors="none", edgecolors=RED,
               linewidths=1.6, label="支持向量")
    w = clf.coef_[0]
    unit = w / np.linalg.norm(w)
    base = np.array([0.0, -(clf.intercept_[0]) / w[1]])
    tip = base + unit * (2.0 / np.linalg.norm(w))
    ax.annotate("", xy=tip, xytext=base,
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.7))
    ax.annotate(r"$\dfrac{2}{\|\mathbf{w}\|}$", xy=(base + tip) / 2, xytext=(9, 12),
                textcoords="offset points", fontsize=13, color=RED)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("最大间隔超平面与支持向量")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig2_svm_margin.jpg"))


# ---------------------------------------------------------------------------
# 2. 软间隔与惩罚系数 C
# ---------------------------------------------------------------------------
def fig_soft_margin_C():
    rs.apply()
    X, y = two_gaussians(n=40, sep=0.55, seed=7)
    yy_ = np.where(y == 0, -1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    lo, hi = X.min(0) - 0.8, X.max(0) + 0.8
    for ax, C in zip(axes, (0.05, 100.0)):
        clf = SVC(kernel="linear", C=C).fit(X, yy_)
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 350), np.linspace(lo[1], hi[1], 350))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#fae3d0", "#d9e6f5"], alpha=0.5)
        ax.contourf(xx, yy, Z, levels=[-1, 1], colors=["#ffe9c9"], alpha=0.55)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[RED, "#333333", RED],
                   linewidths=[1.2, 1.9, 1.2], linestyles=["--", "-", "--"])
        scatter2(ax, X, y, s=26)
        viol = int((yy_ * clf.decision_function(X) < 1).sum())
        ax.set_title(f"$C={C:g}$：间隔宽度 {2 / np.linalg.norm(clf.coef_[0]):.2f}，"
                     f"违反间隔的样本 {viol} 个")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.grid(alpha=0.35)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_soft_margin_C.jpg"))


# ---------------------------------------------------------------------------
# 3. 核技巧：低维线性不可分 → 高维线性可分
# ---------------------------------------------------------------------------
def fig_kernel_trick():
    rs.apply()
    rng = np.random.default_rng(11)
    n = 90
    r0 = rng.uniform(0.0, 0.75, n)
    r1 = rng.uniform(1.25, 2.0, n)
    t0 = rng.uniform(0, 2 * np.pi, n)
    t1 = rng.uniform(0, 2 * np.pi, n)
    Xc = np.vstack([np.c_[r0 * np.cos(t0), r0 * np.sin(t0)],
                    np.c_[r1 * np.cos(t1), r1 * np.sin(t1)]])
    yc = np.r_[np.zeros(n, int), np.ones(n, int)]
    Zc = Xc[:, 0] ** 2 + Xc[:, 1] ** 2

    fig = plt.figure(figsize=(9.6, 4.0))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], s=20, c=ORANGE, edgecolors="white", linewidths=0.3, label="类别 0")
    ax.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], s=20, c=BLUE, edgecolors="white", linewidths=0.3, label="类别 1")
    xs = np.linspace(-2.2, 2.2, 10)
    ax.plot(xs, -0.35 * xs + 0.55, color="#333333", lw=1.8, ls="--", label="线性边界（失效）")
    ax.set_title("（a）原始二维空间：线性不可分")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.35)
    ax.set_aspect("equal")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], Zc[yc == 0], s=14, c=ORANGE, alpha=0.9)
    ax3.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], Zc[yc == 1], s=14, c=BLUE, alpha=0.9)
    gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 12), np.linspace(-2.2, 2.2, 12))
    ax3.plot_surface(gx, gy, np.full_like(gx, 1.0), alpha=0.28, color=RED, linewidth=0)
    ax3.set_title("（b）映射到三维：可用平面分开")
    ax3.set_xlabel(r"$x_1$")
    ax3.set_ylabel(r"$x_2$")
    ax3.set_zlabel(r"$z=x_1^2+x_2^2$")
    ax3.view_init(elev=22, azim=-58)
    ax3.set_zlim(0, 4.6)
    ax3.set_box_aspect((1, 1, 0.62))
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_kernel_trick.jpg"))


# ---------------------------------------------------------------------------
# 4/5. 核函数对比与 gamma 的影响（真实 Auto 数据，二维特征）
# ---------------------------------------------------------------------------
def fig_kernel_compare():
    rs.apply()
    X, y = auto_2d()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
    cfg = [("线性核 linear", dict(kernel="linear")),
           ("多项式核 poly (d=3)", dict(kernel="poly", degree=3)),
           ("高斯核 RBF", dict(kernel="rbf", C=1.0, gamma="scale"))]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
    lo, hi = X.min(0) - 0.4, X.max(0) + 0.4
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 350), np.linspace(lo[1], hi[1], 350))
    for ax, (name, kw) in zip(axes, cfg):
        clf = SVC(**kw).fit(Xtr, ytr)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#fae3d0", "#d9e6f5"], alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.0], colors=["#333333"], linewidths=1.8)
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=14, c=ORANGE, alpha=0.8,
                   edgecolors="none", label="训练集")
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=14, c=BLUE, alpha=0.8,
                   edgecolors="none")
        ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=34, c=ORANGE,
                   edgecolors="white", linewidths=0.5, marker="^")
        ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=34, c=BLUE,
                   edgecolors="white", linewidths=0.5, marker="^")
        ax.set_title(f"{name}\n测试集准确率 {clf.score(Xte, yte) * 100:.2f}%", fontsize=10.5)
        ax.set_xlabel("weight（标准化）")
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("horsepower（标准化）")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_kernel_compare.jpg"))


def fig_gamma_effect():
    rs.apply()
    X, y = auto_2d()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
    lo, hi = X.min(0) - 0.4, X.max(0) + 0.4
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 350), np.linspace(lo[1], hi[1], 350))
    for ax, g in zip(axes, (0.05, 0.5, 5.0)):
        clf = SVC(kernel="rbf", C=1.0, gamma=g).fit(Xtr, ytr)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#fae3d0", "#d9e6f5"], alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.0], colors=["#333333"], linewidths=1.8)
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=12, c=ORANGE, alpha=0.75, edgecolors="none")
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=12, c=BLUE, alpha=0.75, edgecolors="none")
        ax.set_title(f"$\\gamma={g:g}$　训练集 {clf.score(Xtr, ytr) * 100:.1f}% / "
                     f"测试集 {clf.score(Xte, yte) * 100:.1f}%", fontsize=10)
        ax.set_xlabel("weight（标准化）")
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("horsepower（标准化）")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_gamma_effect.jpg"))


# ---------------------------------------------------------------------------
# 6. 感知机的更新过程与收敛性
# ---------------------------------------------------------------------------
def fig_perceptron_update():
    rs.apply()
    df = load_diabetes()
    feats = ["Glucose", "BMI"]
    X = StandardScaler().fit_transform(df[feats].values)
    y = df["Outcome"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)

    clf = Perceptron(max_iter=1, tol=None, shuffle=False, random_state=RS)
    snaps, errs = {}, []
    for ep in range(1, 51):
        clf.partial_fit(Xtr, ytr, classes=np.array([0, 1]))
        errs.append(float((clf.predict(Xtr) != ytr).mean()) * 100)
        if ep in (1, 3, 10, 50):
            snaps[ep] = clf.coef_[0].copy(), float(clf.intercept_[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=12, c=ORANGE, alpha=0.55,
               edgecolors="none", label="未患病")
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=12, c=BLUE, alpha=0.55,
               edgecolors="none", label="患病")
    lo, hi = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    xs = np.linspace(lo, hi, 200)
    for k, (ep, (w, b)) in enumerate(snaps.items()):
        ax.plot(xs, -(w[0] * xs + b) / w[1], lw=1.6, alpha=0.9,
                color=plt.cm.viridis(k / 3), label=f"第 {ep} 轮")
    ax.set_xlim(lo, hi)
    ax.set_ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
    ax.set_xlabel("Glucose（标准化）")
    ax.set_ylabel("BMI（标准化）")
    ax.set_title("（a）感知机决策边界的迭代过程")
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(alpha=0.35)

    ax = axes[1]
    ax.plot(range(1, 51), errs, color=BLUE, lw=2, marker="o", ms=3)
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("训练集错分率 / %")
    ax.set_title("（b）训练错分率随迭代变化（数据线性不可分，不收敛到 0）")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_perceptron_update.jpg"))


# ---------------------------------------------------------------------------
# 7. 感知机与 SVM 的间隔差异
# ---------------------------------------------------------------------------
def fig_perceptron_vs_svm():
    rs.apply()
    df = load_diabetes()
    X = StandardScaler().fit_transform(df[["Glucose", "BMI"]].values)
    y = df["Outcome"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)

    perc = Perceptron(max_iter=1000, tol=1e-3, random_state=RS).fit(Xtr, ytr)
    svm = SVC(kernel="linear", C=1.0).fit(Xtr, ytr)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharex=True, sharey=True)
    lo, hi = X.min(0) - 0.3, X.max(0) + 0.3
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 350), np.linspace(lo[1], hi[1], 350))
    for ax, clf, title, lv in (
            (axes[0], perc, f"感知机（测试集 {perc.score(Xte, yte) * 100:.2f}%）", [0.0]),
            (axes[1], svm, f"线性 SVM（测试集 {svm.score(Xte, yte) * 100:.2f}%）", [-1.0, 0.0, 1.0])):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#fae3d0", "#d9e6f5"], alpha=0.55)
        if len(lv) == 3:                      # SVM：把间隔带浅色填充
            ax.contourf(xx, yy, Z, levels=[-1, 1], colors=["#ffe9c9"], alpha=0.5)
        ax.contour(xx, yy, Z, levels=lv, colors=[RED if v else "#333333" for v in lv],
                   linewidths=[1.2 if v else 1.9 for v in lv],
                   linestyles=["--" if v else "-" for v in lv])
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=12, c=ORANGE, alpha=0.6, edgecolors="none")
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=12, c=BLUE, alpha=0.6, edgecolors="none")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Glucose（标准化）")
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("BMI（标准化）")
    sv = svm.support_
    axes[1].scatter(Xtr[sv[:10], 0], Xtr[sv[:10], 1], s=110, facecolors="none",
                    edgecolors=RED, linewidths=1.4)
    axes[1].annotate(f"红圈为部分支持向量（共 {len(sv)} 个），浅色带为间隔区域",
                     xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8.5, color="#333333")
    print(f"    [信息] 线性 SVM 支持向量个数：{len(sv)}")
    axes[0].annotate("边界仅要求分开样本，未最大化间隔", xy=(0.03, 0.03), xycoords="axes fraction",
                     fontsize=8.5, color="#333333")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_perceptron_vs_svm.jpg"))


# ---------------------------------------------------------------------------
# 8/9/10. 真实数据相关：中位数二分、类别分布、参数网格
# ---------------------------------------------------------------------------
def fig_mpg_binning():
    rs.apply()
    df = load_auto()
    med = df["mpg"].median()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bins = np.arange(df["mpg"].min(), df["mpg"].max() + 2, 2)
    ax.hist(df.loc[df["grade"] == 0, "mpg"], bins=bins, color=ORANGE, alpha=0.85, label="grade = 0（mpg < 中位数）")
    ax.hist(df.loc[df["grade"] == 1, "mpg"], bins=bins, color=BLUE, alpha=0.85, label="grade = 1（mpg ≥ 中位数）")
    ax.axvline(med, color=RED, lw=2, ls="--", label=f"中位数 = {med:g}")
    ax.annotate(f"两侧各 {int((df['grade'] == 0).sum())} 条样本", xy=(med, 0.92),
                xycoords=("data", "axes fraction"), xytext=(-95, 0), textcoords="offset points",
                fontsize=9.5, color="#333333")
    ax.set_xlabel("mpg（每加仑英里数）")
    ax.set_ylabel("样本数")
    ax.set_title("按 mpg 中位数构造二分类标签")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig2_mpg_binning.jpg"))


def fig_class_balance():
    rs.apply()
    df = load_diabetes()
    vc = df["Outcome"].value_counts().sort_index()
    base = vc.max() / vc.sum() * 100
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    bars = ax.bar(["未患糖尿病（0）", "患糖尿病（1）"], vc.values, color=[ORANGE, BLUE],
                  width=0.55, edgecolor="white")
    for rect, v in zip(bars, vc.values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 8, f"{v}（{v / vc.sum() * 100:.1f}%）",
                ha="center", fontsize=10)
    ax.axhline(base, color=RED, ls="--", lw=1.4, label=f"多数类基线 {base:.1f}%")
    ax.axhline(72.73, color=GREEN, ls=":", lw=1.6, label="感知机准确率 72.73%")
    ax.set_ylim(0, 620)
    ax.set_ylabel("样本数")
    ax.set_title("糖尿病数据集的类别分布与“不学习”基线")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    rs.save(fig, os.path.join(OUT, "fig2_class_balance.jpg"))


def fig_param_grid():
    rs.apply()
    df = load_auto()
    feats = ["cylinders", "displacement", "horsepower", "weight", "acceleration",
             "model_year", "origin"]
    X = StandardScaler().fit_transform(df[feats].values)
    y = df["grade"].values
    Cs = [0.1, 1.0, 10.0, 100.0]
    gs = [0.01, 0.05, 0.143, 0.5, 2.0]
    acc = np.zeros((len(gs), len(Cs)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    for i, g in enumerate(gs):
        for j, C in enumerate(Cs):
            scores = []
            for tr, te in cv.split(X, y):
                m = SVC(kernel="rbf", C=C, gamma=g).fit(X[tr], y[tr])
                scores.append(m.score(X[te], y[te]))
            acc[i, j] = np.mean(scores) * 100

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    im = ax.imshow(acc, cmap="YlGnBu", aspect="auto", vmin=acc.min(), vmax=acc.max())
    ax.set_xticks(range(len(Cs)), [f"{c:g}" for c in Cs])
    ax.set_yticks(range(len(gs)), [f"{g:g}" for g in gs])
    ax.set_xlabel(r"惩罚系数 $C$")
    ax.set_ylabel(r"核参数 $\gamma$")
    for i in range(len(gs)):
        for j in range(len(Cs)):
            ax.text(j, i, f"{acc[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if acc[i, j] > acc.mean() + 1.2 else "#1a1a1a")
    bi, bj = np.unravel_index(np.argmax(acc), acc.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2.2))
    ax.set_title(f"5 折交叉验证准确率（%）　最优：C={Cs[bj]:g}，γ={gs[bi]:g}，{acc[bi, bj]:.1f}%",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, label="准确率 / %")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig2_param_grid.jpg"))


# ===========================================================================
# 公式
# ===========================================================================
FORMULAS = {
    # ---- SVM 原理 ----
    "f2_hyperplane": r"$\mathbf{w}^{\mathrm{T}}\mathbf{x}+b=0$",
    "f2_margin": r"$\mathrm{margin}=\dfrac{2}{\|\mathbf{w}\|}$",
    "f2_hard": r"$\min_{\mathbf{w},b}\;\dfrac{1}{2}\|\mathbf{w}\|^2\quad \mathrm{s.t.}\;\; y_i\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b\right)\geq 1,\; i=1,\dots,m$",
    "f2_soft": r"$\min_{\mathbf{w},b,\xi}\;\dfrac{1}{2}\|\mathbf{w}\|^2+C\sum_{i=1}^{m}\xi_i\quad \mathrm{s.t.}\;\; y_i\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b\right)\geq 1-\xi_i,\;\xi_i\geq 0$",
    "f2_lagrange": r"$L(\mathbf{w},b,\alpha)=\dfrac{1}{2}\|\mathbf{w}\|^2-\sum_{i=1}^{m}\alpha_i\left[y_i\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b\right)-1\right],\qquad \alpha_i\geq 0$",
    "f2_dual": r"$\max_{\alpha}\;\sum_{i=1}^{m}\alpha_i-\dfrac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy_iy_j\mathbf{x}_i^{\mathrm{T}}\mathbf{x}_j\quad \mathrm{s.t.}\;\;\sum_{i=1}^{m}\alpha_iy_i=0,\;0\leq\alpha_i\leq C$",
    "f2_kkt": r"$\alpha_i\left[y_i\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b\right)-1+\xi_i\right]=0,\qquad \mathbf{w}=\sum_{i=1}^{m}\alpha_iy_i\mathbf{x}_i$",
    "f2_decision": r"$f(\mathbf{x})=\mathrm{sign}\left(\sum_{i=1}^{m}\alpha_iy_iK\left(\mathbf{x}_i,\mathbf{x}\right)+b\right)$",
    "f2_hinge": r"$\ell_{\mathrm{hinge}}(z)=\max\left(0,\,1-z\right),\qquad z=y\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}+b\right)$",
    # ---- 核函数 ----
    "f2_kernel_def": r"$K\left(\mathbf{x},\mathbf{z}\right)=\phi(\mathbf{x})^{\mathrm{T}}\phi(\mathbf{z})$",
    "f2_rbf": r"$K\left(\mathbf{x},\mathbf{z}\right)=\exp\left(-\gamma\left\|\mathbf{x}-\mathbf{z}\right\|^2\right),\qquad \gamma=\dfrac{1}{2\sigma^2}>0$",
    "f2_poly": r"$K\left(\mathbf{x},\mathbf{z}\right)=\left(\mathbf{x}^{\mathrm{T}}\mathbf{z}+c\right)^d$",
    "f2_scale": r"$x'=\dfrac{x-\mu}{\sigma}$",
    # ---- 感知机 ----
    "f2_perc_model": r"$f(\mathbf{x})=\mathrm{sign}\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}+b\right)$",
    "f2_perc_loss": r"$L(\mathbf{w},b)=-\sum_{\mathbf{x}_i\in M}y_i\left(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b\right)$",
    "f2_perc_grad": r"$\nabla_{\mathbf{w}}L=-\sum_{\mathbf{x}_i\in M}y_i\mathbf{x}_i,\qquad \nabla_{b}L=-\sum_{\mathbf{x}_i\in M}y_i$",
    "f2_perc_update": r"$\mathbf{w}\leftarrow\mathbf{w}+\eta\,y_i\mathbf{x}_i,\qquad b\leftarrow b+\eta\,y_i$",
    "f2_novikoff": r"$k\leq\left(\dfrac{R}{\gamma}\right)^2,\qquad \gamma=\min_i y_i\left(\mathbf{w}^{*\mathrm{T}}\mathbf{x}_i+b^{*}\right),\;\;R=\max_i\|\mathbf{x}_i\|$",
    # ---- 评估指标 ----
    "f2_acc": r"$acc=\dfrac{TP+TN}{TP+TN+FP+FN}$",
    "f2_prf": r"$P=\dfrac{TP}{TP+FP},\qquad R=\dfrac{TP}{TP+FN},\qquad F1=\dfrac{2PR}{P+R}$",
    "f2_macro": r"$F1_{\mathrm{macro}}=\dfrac{1}{K}\sum_{k=1}^{K}F1_k,\qquad F1_{\mathrm{weighted}}=\sum_{k=1}^{K}\dfrac{n_k}{N}F1_k$",
    "f2_baseline": r"$acc_{\mathrm{baseline}}=\max_k\dfrac{n_k}{N}$",
}


def render_formulas():
    print("渲染公式 ...")
    rs.apply()
    out_dir = os.path.join(rs.ASSETS, "exp2", "formula")
    os.makedirs(out_dir, exist_ok=True)
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=20, color="black")
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300,
                    bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
    print(f"  共 {len(FORMULAS)} 个公式 -> 素材/exp2/formula/")


if __name__ == "__main__":
    print("绘制实验二理论原理图 ...")
    fig_svm_margin()
    fig_soft_margin_C()
    fig_kernel_trick()
    fig_kernel_compare()
    fig_gamma_effect()
    fig_perceptron_update()
    fig_perceptron_vs_svm()
    fig_mpg_binning()
    fig_class_balance()
    fig_param_grid()
    render_formulas()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
