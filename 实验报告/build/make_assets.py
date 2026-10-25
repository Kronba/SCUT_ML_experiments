# -*- coding: utf-8 -*-
"""实验一报告素材生成（v2）：理论原理图 + 公式图。

v2 改动：不再从课件 PDF 截图（观感差、带模板装饰与裁切），
统一改为按教材理论用 matplotlib 自绘；所有插图走 report_style 主题、600 dpi 白底。
    * 概念图（混淆矩阵 / P-R / ROC / 交叉验证）与结果图见 redraw_concepts.py
    * notebook 白底输出图见 redraw_notebook.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

import report_style as rs

ASSETS = rs.ASSETS
os.makedirs(os.path.join(ASSETS, "theory"), exist_ok=True)
os.makedirs(os.path.join(ASSETS, "formula"), exist_ok=True)

BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED
save = rs.save


# ===========================================================================
# 一、理论原理图（自绘）
# ===========================================================================
def fig_sigmoid():
    """Sigmoid 函数与对数几率变换。"""
    rs.apply()
    z = np.linspace(-6, 6, 500)
    sig = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))

    ax = axes[0]
    ax.plot(z, sig, color=BLUE, lw=2)
    ax.axhline(0.5, color=RED, ls="--", lw=1)
    ax.axvline(0, color="#888888", ls=":", lw=1)
    ax.plot(0, 0.5, "o", color=RED, ms=5)
    ax.annotate(r"$\sigma(0)=0.5$", xy=(0, 0.5), xytext=(0.8, 0.32),
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9), color=RED)
    ax.text(-5.8, 0.16, "饱和区\n（梯度≈0）", fontsize=8.5, color="#555555")
    ax.text(5.8, 0.74, "饱和区\n（梯度≈0）", fontsize=8.5, color="#555555", ha="right")
    ax.set_xlabel(r"$z=\mathbf{w}^{\mathrm{T}}\mathbf{x}+b$")
    ax.set_ylabel(r"$\sigma(z)=P(y=1\,|\,\mathbf{x})$")
    ax.set_title("（a）Sigmoid 函数：把实数轴压缩到 (0, 1)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    ax = axes[1]
    p = np.linspace(0.01, 0.99, 400)
    ax.plot(p, np.log(p / (1 - p)), color=GREEN, lw=2)
    ax.axhline(0, color="#888888", ls=":", lw=1)
    ax.axvline(0.5, color=RED, ls="--", lw=1)
    ax.set_xlabel(r"概率 $p$")
    ax.set_ylabel(r"对数几率 $\ln\frac{p}{1-p}$")
    ax.set_title("（b）对数几率是 $z$ 的线性函数（logit 变换）")
    ax.set_ylim(-4.8, 4.8)
    ax.grid(True)

    fig.tight_layout()
    save(fig, os.path.join(ASSETS, "theory", "fig_sigmoid.jpg"))


def fig_logloss():
    """对数损失（负对数似然）随 z 的变化。"""
    rs.apply()
    z = np.linspace(-4, 4, 500)
    sig = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(z, -np.log(sig), color=BLUE, lw=2, label=r"$y=1:\;-\ln\sigma(z)$")
    ax.plot(z, -np.log(1 - sig), color=ORANGE, lw=2, label=r"$y=0:\;-\ln(1-\sigma(z))$")
    ax.axvline(0, color="#888888", ls=":", lw=1)
    ax.set_xlabel(r"$z=\mathbf{w}^{\mathrm{T}}\mathbf{x}+b$")
    ax.set_ylabel("损失")
    ax.set_title("对数损失的凸性与惩罚机制")
    ax.set_ylim(0, 4.2)
    ax.legend(loc="upper center")
    ax.grid(True)
    save(fig, os.path.join(ASSETS, "theory", "fig_logloss.jpg"))


def fig_boundary_geo():
    """决策边界的几何意义：真实 Simmons 数据 + 最大似然解。"""
    rs.apply()
    from sklearn.linear_model import LogisticRegression
    X, y = rs.simmons()
    clf = LogisticRegression().fit(X, y)          # 与 notebook 完全一致的拟合
    w, b = clf.coef_[0], clf.intercept_[0]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xs = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 300)
    ys = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 300)
    XX, YY = np.meshgrid(xs, ys)
    P = 1 / (1 + np.exp(-(w[0] * XX + w[1] * YY + b)))
    ax.contourf(XX, YY, P, levels=[0, 0.5, 1], colors=["#fae3d0", "#d9e6f5"], alpha=0.8)
    ax.contour(XX, YY, P, levels=[0.5], colors=[RED], linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=26, c=ORANGE, edgecolors="white",
               linewidths=0.5, label="Coupon = 0（不使用优惠券）")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=26, c=BLUE, edgecolors="white",
               linewidths=0.5, label="Coupon = 1（使用优惠券）")
    px = np.mean(xs)
    base = np.array([px, -(b + w[0] * px) / w[1]])
    unit = w / np.linalg.norm(w)
    tip = base + 0.75 * unit
    ax.annotate("", xy=tip, xytext=base,
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.7))
    ax.annotate(r"$\mathbf{w}$（法向量）", xy=tip, xytext=(7, 6), textcoords="offset points",
                color="#333333")
    ax.text(0.025, 0.06, r"边界：$\mathbf{w}^{\mathrm{T}}\mathbf{x}+b=0$，即 $P(y=1)=0.5$",
            transform=ax.transAxes, color=RED, fontsize=9.5)
    ax.set_xlabel("Spending（消费额）")
    ax.set_ylabel("Card（持卡情况）")
    ax.set_title("Logistic 回归决策边界的几何解释（Simmons 数据实测）")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)
    save(fig, os.path.join(ASSETS, "theory", "fig_boundary_geo.jpg"))


def draw_cov_ellipse(ax, pts, mu, color, n_std=1.5):
    from matplotlib.patches import Ellipse
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mu, width, height, angle=angle, fill=False,
                         edgecolor=color, lw=1.3, ls="--", alpha=0.9))


def fig_lda_geo():
    """Fisher 判别几何：类内/类间散度与最优投影方向。"""
    rs.apply()
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data[:, 2:4]
    y = iris.target
    m = y < 2
    X, y = X[m], y[m]

    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    Sw = sum((X[y == c] - X[y == c].mean(0)).T @ (X[y == c] - X[y == c].mean(0))
             for c in (0, 1))
    w = np.linalg.solve(Sw, mu0 - mu1)
    w = w / np.linalg.norm(w)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax = axes[0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=24, c=BLUE, edgecolors="white",
               linewidths=0.4, label="山鸢尾 (setosa)")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=24, c=ORANGE, edgecolors="white",
               linewidths=0.4, label="变色鸢尾 (versicolor)")
    for mu, c, cls in ((mu0, BLUE, 0), (mu1, ORANGE, 1)):
        ax.plot(*mu, "*", ms=15, color=c, markeredgecolor="#333333",
                markeredgewidth=0.6, zorder=5)
        draw_cov_ellipse(ax, X[y == cls], mu, c)
    mu_all = X.mean(0)
    ax.annotate("", xy=mu0, xytext=mu1,
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6, shrinkA=2, shrinkB=2))
    ax.annotate(r"$\mu_0-\mu_1$", xy=(mu0 + mu1) / 2, xytext=(2, -14),
                textcoords="offset points", color="#333333")
    tt = np.linspace(-8, 8, 2)
    line = mu_all[None, :] + tt[:, None] * w[None, :]
    ax.plot(line[:, 0], line[:, 1], color=RED, lw=1.2, ls=":", zorder=4)
    tip = mu_all + 1.5 * w
    ax.annotate("", xy=tip, xytext=mu_all,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.annotate(r"$\mathbf{w}=S_w^{-1}(\mu_0-\mu_1)$", xy=tip, xytext=(6, 8),
                textcoords="offset points", color=RED)
    ax.set_xlabel("花瓣长度 / cm")
    ax.set_ylabel("花瓣宽度 / cm")
    ax.set_title("（a）类内散度椭圆与 Fisher 投影方向")
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.7)
    ax.set_ylim(X[:, 1].min() - 0.35, X[:, 1].max() + 0.45)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    proj = X @ w
    ax.hist(proj[y == 0], bins=10, color=BLUE, alpha=0.65, label="setosa 投影")
    ax.hist(proj[y == 1], bins=10, color=ORANGE, alpha=0.65, label="versicolor 投影")
    ax.axvline(proj[y == 0].mean(), color=BLUE, ls="--", lw=1)
    ax.axvline(proj[y == 1].mean(), color=ORANGE, ls="--", lw=1)
    thr = (proj[y == 0].mean() + proj[y == 1].mean()) / 2
    ax.axvline(thr, color="#333333", ls=":", lw=1.3)
    ax.annotate("判别阈值", xy=(thr, 0), xytext=(6, 30), textcoords="offset points",
                color="#333333", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))
    ax.set_xlabel(r"投影值 $y=\mathbf{w}^{\mathrm{T}}\mathbf{x}$")
    ax.set_ylabel("样本数")
    ax.set_title("（b）投影后类间距离最大化、类内方差最小化")
    ax.legend()
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    save(fig, os.path.join(ASSETS, "theory", "fig_lda_geo.jpg"))


# ===========================================================================
# 二、公式图（mathtext / Computer Modern）
# ===========================================================================
FORMULAS = {
    # ---- 逻辑斯蒂回归 ----
    "f_linear": r"$z=\mathbf{w}^{\mathrm{T}}\mathbf{x}+b=w_1x_1+w_2x_2+\cdots+w_dx_d+b$",
    "f_sigmoid": r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$",
    "f_logit": r"$\ln\dfrac{p}{1-p}=\mathbf{w}^{\mathrm{T}}\mathbf{x}+b$",
    "f_sigmoid_deriv": r"$\sigma'(z)=\dfrac{e^{-z}}{\left(1+e^{-z}\right)^2}=\sigma(z)\left(1-\sigma(z)\right)$",
    "f_likelihood": r"$\ell(\mathbf{w},b)=\sum_{i=1}^{m}\left[y_i\ln\sigma(z_i)+(1-y_i)\ln\left(1-\sigma(z_i)\right)\right]$",
    "f_logloss": r"$L(\mathbf{w},b)=-\ell(\mathbf{w},b)=\sum_{i=1}^{m}\left[-y_i\ln\hat{p}_i-(1-y_i)\ln(1-\hat{p}_i)\right]$",
    "f_grad": r"$\dfrac{\partial \ell}{\partial \mathbf{w}}=\sum_{i=1}^{m}\left(y_i-\sigma(z_i)\right)\mathbf{x}_i,\qquad \dfrac{\partial \ell}{\partial b}=\sum_{i=1}^{m}\left(y_i-\sigma(z_i)\right)$",
    "f_newton": r"$\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}-\mathbf{H}^{-1}\nabla\ell(\mathbf{w}^{(t)}),\qquad \mathbf{H}=\dfrac{\partial^2\ell}{\partial\mathbf{w}\,\partial\mathbf{w}^{\mathrm{T}}}$",
    "f_boundary": r"$\mathbf{w}^{\mathrm{T}}\mathbf{x}+b=0\;\Longleftrightarrow\;\sigma(z)=0.5\;\Longleftrightarrow\;\dfrac{P(y=1\mid\mathbf{x})}{P(y=0\mid\mathbf{x})}=1$",
    # ---- 性能度量 ----
    "f_acc": r"$E(f;D)=\dfrac{1}{m}\sum_{i=1}^{m}\mathrm{I}\left(f(\mathbf{x}_i)\neq y_i\right),\qquad acc=\dfrac{1}{m}\sum_{i=1}^{m}\mathrm{I}\left(f(\mathbf{x}_i)=y_i\right)=1-E(f;D)$",
    "f_prf": r"$P=\dfrac{TP}{TP+FP},\qquad R=\dfrac{TP}{TP+FN},\qquad F1=\dfrac{2\times P\times R}{P+R}$",
    # ---- 线性判别分析 ----
    "f_class_mean": r"$\mu_c=\dfrac{1}{m_c}\sum_{\mathbf{x}_i\in X_c}\mathbf{x}_i,\qquad \mu=\dfrac{1}{m}\sum_{i=1}^{m}\mathbf{x}_i$",
    "f_sw": r"$S_w=\sum_{c}\sum_{\mathbf{x}_i\in X_c}\left(\mathbf{x}_i-\mu_c\right)\left(\mathbf{x}_i-\mu_c\right)^{\mathrm{T}}$",
    "f_sb": r"$S_b=\sum_{c}m_c\left(\mu_c-\mu\right)\left(\mu_c-\mu\right)^{\mathrm{T}}$",
    "f_fisher": r"$J(\mathbf{w})=\dfrac{\mathbf{w}^{\mathrm{T}}S_b\mathbf{w}}{\mathbf{w}^{\mathrm{T}}S_w\mathbf{w}}$",
    "f_fisher_sol": r"$\mathbf{w}^{*}=S_w^{-1}\left(\mu_0-\mu_1\right)$",
    "f_fisher_proof": r"$\dfrac{\partial J}{\partial \mathbf{w}}=0\;\Longrightarrow\;\left(\mathbf{w}^{\mathrm{T}}S_b\mathbf{w}\right)S_w\mathbf{w}=\left(\mathbf{w}^{\mathrm{T}}S_w\mathbf{w}\right)S_b\mathbf{w}\;\Longrightarrow\;S_b\mathbf{w}=\lambda S_w\mathbf{w}$",
    "f_lda_multi": r"$S_b\mathbf{v}=\lambda S_w\mathbf{v},\qquad \mathbf{V}=\left[\mathbf{v}_1,\dots,\mathbf{v}_{K-1}\right],\qquad y=\mathbf{V}^{\mathrm{T}}\mathbf{x}$",
    "f_lda_disc": r"$\delta_k(\mathbf{x})=\mathbf{x}^{\mathrm{T}}\Sigma^{-1}\mu_k-\dfrac{1}{2}\mu_k^{\mathrm{T}}\Sigma^{-1}\mu_k+\ln\pi_k,\qquad \hat{y}=\arg\max_{k}\delta_k(\mathbf{x})$",
    "f_lda_binary_disc": r"$\delta_0(\mathbf{x})-\delta_1(\mathbf{x})=0\;\Longrightarrow\;\mathbf{w}^{\mathrm{T}}\mathbf{x}+b=0$",
    "f_scale": r"$x'=\dfrac{x-\mu}{\sigma},\qquad \mu=\dfrac{1}{m}\sum_{i=1}^{m}x_i,\qquad \sigma=\sqrt{\dfrac{1}{m}\sum_{i=1}^{m}\left(x_i-\mu\right)^2}$",
}


def render_formulas():
    print("渲染公式 ...")
    rs.apply()
    out_dir = os.path.join(ASSETS, "formula")
    for name, tex in FORMULAS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=20, color="black")
        path = os.path.join(out_dir, name + ".png")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
    print(f"  共 {len(FORMULAS)} 个公式 -> formula/")


if __name__ == "__main__":
    print("绘制理论原理图 ...")
    fig_sigmoid()
    fig_logloss()
    fig_boundary_geo()
    fig_lda_geo()
    render_formulas()
    print("完成 ->", os.path.relpath(ASSETS, rs.PROJ_DIR))
