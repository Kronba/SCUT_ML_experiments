# -*- coding: utf-8 -*-
"""实验报告统一绘图主题（v2）。

目标：报告内 13 张插图共用一套风格——白底、中文衬线、Computer Modern 数学字体、
统一配色与线宽，全部以 600 dpi 输出为 jpg（作业命名规范要求内插图片为 jpg）。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
ASSETS = os.path.join(REPORT_DIR, "素材")

# 统一配色
BLUE, ORANGE, GREEN, RED = "#2f6fb5", "#e07b39", "#3f8f5a", "#c0392b"
GRAY, LIGHT, GRID = "#666666", "#f2f2f2", "#d9d9d9"

FONT_CN = "Noto Serif CJK SC"
DPI = 600


def apply():
    """把统一主题写入全局 rcParams。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_CN, "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.dpi": 200,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
    })


def save(fig, path):
    """按统一规格保存（600 dpi、白底、紧凑边距），并打印文件大小。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = {}
    if path.lower().endswith((".jpg", ".jpeg")):
        kw["pil_kwargs"] = {"quality": 92, "optimize": True}
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.06,
                facecolor="white", **kw)
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path, PROJ_DIR):58s} {os.path.getsize(path)//1024:>4d} KB")


def simmons():
    """读取 Simmons 数据集，返回特征矩阵与标签（与 notebook 完全一致）。"""
    import pandas as pd
    data = pd.read_excel(os.path.join(PROJ_DIR, "data", "exp1", "Simmons.xlsx"))
    return data[["Spending", "Card"]].values, data["Coupon"].values
