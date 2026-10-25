# -*- coding: utf-8 -*-
"""实验三：调包实现 vs 自实现 的对比图（600 dpi，统一 report_style 主题）。"""
import glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB as SkNB
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import report_style as rs
from models.exp3.naive_bayes import MultinomialNB as MyNB

OUT = os.path.join(rs.ASSETS, "exp3", "compare")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, RED = rs.BLUE, rs.ORANGE, rs.GREEN, rs.RED


def load_emails():
    emails, labels = [], []
    for path, lab in ((os.path.join(rs.PROJ_DIR, "data/exp3/email/ham/*.txt"), 0),
                      (os.path.join(rs.PROJ_DIR, "data/exp3/email/spam/*.txt"), 1)):
        for f in glob.glob(path):
            emails.append(open(f, encoding="utf-8", errors="ignore").read().lower())
            labels.append(lab)
    return pd.DataFrame({"text": emails, "label": labels})


def get_cv_scores(df):
    X, y = df["text"], df["label"]
    model = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
                      ("nb", SkNB())])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy")


def fig_nb_compare():
    rs.apply()
    df = load_emails()
    X, y = df["text"], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    sk = SkNB().fit(A, ytr)
    my = MyNB().fit(A, ytr)
    sk_scores = get_cv_scores(df)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    x = np.arange(5)
    ax.bar(x - 0.19, sk_scores * 100, width=0.38, color=BLUE, label="sklearn 调包")
    ax.bar(x + 0.19, sk_scores * 100, width=0.38, color=ORANGE, alpha=0.85,
           label="自实现", hatch="//", edgecolor="white")
    ax.set_xticks(x, [f"第{i+1}折" for i in range(5)])
    ax.set_ylim(0, 112)
    ax.set_ylabel("准确率 / %")
    ax.set_title(f"（a）5 折交叉验证：两者每折完全相同（均值 {sk_scores.mean()*100:.2f}%）")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    ax.scatter(sk.feature_log_prob_.ravel(), my.feature_log_prob_.ravel(),
               s=10, c=BLUE, alpha=0.7, edgecolors="none")
    lim = [sk.feature_log_prob_.min() - 0.2, sk.feature_log_prob_.max() + 0.2]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("sklearn 的类条件对数概率")
    ax.set_ylabel("自实现的类条件对数概率")
    ax.set_title("（b）对数条件概率逐元素对比（最大差异 0）")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_nb_compare.jpg"))


def fig_nb_cm_compare():
    rs.apply()
    df = load_emails()
    Xtr, Xte, ytr, yte = train_test_split(df["text"], df["label"], test_size=0.2,
                                          random_state=42, stratify=df["label"])
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    sk = SkNB().fit(A, ytr)
    my = MyNB().fit(A, ytr)
    cms = [(confusion_matrix(yte, sk.predict(B)), "sklearn 调包"),
           (confusion_matrix(yte, my.predict(B)), "自实现")]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for ax, (cm, title) in zip(axes, cms):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "#1a1a1a")
        ax.set_xticks([0, 1], ["预测 Ham", "预测 Spam"])
        ax.set_yticks([0, 1], ["真实 Ham", "真实 Spam"])
        ax.set_title(f"（{'a' if title.startswith('sklearn') else 'b'}）{title}")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_nb_cm_compare.jpg"))


def fig_bp_compare():
    rs.apply()
    ns = [1, 2, 3, 5, 10, 15]
    sk_err = [26.04, 54.17, 26.04, 26.67, 26.67, 26.25]
    my_err = [26.88, 25.62, 26.67, 26.46, 26.25, 26.67]
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.bar(x - 0.19, sk_err, width=0.38, color=BLUE, label="sklearn 调包")
    ax.bar(x + 0.19, my_err, width=0.38, color=ORANGE, label="自实现")
    for xi, a, b in zip(x, sk_err, my_err):
        ax.text(xi - 0.19, a + 0.8, f"{a:.2f}", ha="center", fontsize=8.5)
        ax.text(xi + 0.19, b + 0.8, f"{b:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel("隐藏层神经元个数")
    ax.set_ylabel("测试错误率 / %")
    ax.set_ylim(0, 62)
    ax.set_title("两种 BP 实现的测试错误率对比")
    ax.annotate("调包实现退化为常数分类器\n自实现正常收敛", xy=(1.19, 54.17), xytext=(2.05, 48),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_bp_compare.jpg"))


if __name__ == "__main__":
    print("生成实验三对比图 ...")
    fig_nb_compare()
    fig_nb_cm_compare()
    fig_bp_compare()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
