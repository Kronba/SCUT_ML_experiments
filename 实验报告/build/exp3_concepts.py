# -*- coding: utf-8 -*-
"""实验三素材（二）：自绘结果图。

notebook 的贝叶斯部分使用 glob.glob() 读取邮件且未排序，样本顺序取决于文件系统，
重新执行时交叉验证划分会改变，故混淆矩阵图按 notebook 中保存的数值自绘，保证图文一致。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import report_style as rs

OUT = os.path.join(rs.ASSETS, "exp3", "concept")
os.makedirs(OUT, exist_ok=True)


def fig_nb_confusion():
    rs.apply()
    cm = np.array([[5, 0], [0, 5]])          # notebook cell 17 保存的实测混淆矩阵
    labels = ["Ham（正常邮件）", "Spam（垃圾邮件）"]
    acc = np.trace(cm) / cm.sum() * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    ax.set_xticks([0, 1], [f"预测为\n{l}" for l in labels])
    ax.set_yticks([0, 1], [f"真实为\n{l}" for l in labels])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({v / cm.sum() * 100:.1f}%)", ha="center", va="center",
                    fontsize=12, color="white" if v > cm.max() * 0.6 else "#1a1a1a")
    ax.set_title(f"朴素贝叶斯在邮件测试集上的混淆矩阵\n"
                 f"准确率 {acc:.2f}%　查准率 1.00　查全率 1.00　F1 1.00", fontsize=10.5)
    fig.colorbar(im, ax=ax, label="样本数")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_nb_confusion.jpg"))


def fig_bp_confusion():
    """BP 网络（隐藏层 2 个神经元）在红酒测试集上的混淆矩阵。

    由 notebook 的测试误差率 54.17% 与测试集规模 480 反推不出唯一矩阵，
    故这里直接复现 notebook 的划分与训练，重新计算混淆矩阵用于图示。
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import confusion_matrix

    rs.apply()
    path = os.path.join(rs.PROJ_DIR, "data", "exp3", "winequality-red.csv")
    df = pd.read_csv(path, sep=";")
    df["label"] = df["quality"].apply(lambda x: 0 if x <= 5 else 1)
    feats = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
             "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feats], df["label"], test_size=0.3, random_state=42, stratify=df["label"])
    sc = MinMaxScaler()
    Xtr, Xte = sc.fit_transform(X_train), sc.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(2,), activation="relu", solver="adam",
                          max_iter=1000, random_state=42).fit(Xtr, y_train)
    cm = confusion_matrix(y_test, model.predict(Xte))
    err = (1 - np.trace(cm) / cm.sum()) * 100

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    ax.set_xticks([0, 1], ["预测为\nbad", "预测为\ngood"])
    ax.set_yticks([0, 1], ["真实为\nbad", "真实为\ngood"])
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n({v / cm.sum() * 100:.1f}%)", ha="center", va="center",
                    fontsize=12, color="white" if v > cm.max() * 0.6 else "#1a1a1a")
    ax.set_title(f"隐藏层 2 个神经元的 BP 网络混淆矩阵　测试误差率 {err:.2f}%", fontsize=10.5)
    fig.colorbar(im, ax=ax, label="样本数")
    fig.tight_layout()
    rs.save(fig, os.path.join(OUT, "fig3_bp_confusion.jpg"))
    print(f"    [信息] 隐藏层 2 个神经元：测试误差率 {err:.2f}%，混淆矩阵 {cm.tolist()}")


if __name__ == "__main__":
    print("生成实验三自绘结果图 ...")
    fig_nb_confusion()
    fig_bp_confusion()
    print("完成 ->", os.path.relpath(OUT, rs.PROJ_DIR))
