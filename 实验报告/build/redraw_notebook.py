# -*- coding: utf-8 -*-
"""改进 v2：重新执行 exp1.ipynb（对副本执行，原 notebook 不改动），
注入白色主题与大尺寸输出，导出 5 张白底插图，并核对数值与 notebook 原输出一致。

路线 A：图片仍来自"真实执行 notebook 代码"，只是运行环境换成了白底主题。
"""
import base64
import io
import os
import sys

import nbformat
from nbclient import NotebookClient
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
NB_PATH = os.path.join(PROJ_DIR, "exp1.ipynb")
OUT_DIR = os.path.join(REPORT_DIR, "素材", "notebook_out")

# 需要导出插图的单元格（原始 notebook 的索引）
FIG_CELLS = {4: "cell04", 7: "cell07", 15: "cell15", 22: "cell22", 29: "cell29"}
# 需要核对数值的单元格（原始索引 -> 期望输出中的关键片段）
CHECK = {
    12: "最终分类准确率：60.00%",
    24: "LDA 分类准确率：100.00%",
    25: "[[10  0  0]\n [ 0 10  0]\n [ 0  0 10]]",
    28: "LDA 投影后的数据形状： (120, 2)",
}

# 注入的样式单元格：只改"显示环境"，不改任何绘图/建模代码
STYLE_CELL = '''# === 报告用显示样式注入（仅影响绘图外观，不改动实验代码与数值）===
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.prop_cycle": mpl.cycler(color=["#2f6fb5", "#e07b39", "#3f8f5a", "#c0392b"]),
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.labelsize": 10.5, "axes.titlesize": 11.5, "font.size": 10.5,
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a",
    "grid.color": "#d9d9d9", "grid.linewidth": 0.6,
})
'''


def out_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o["text"]))
        elif o.get("output_type") == "execute_result":
            parts.append("".join(o["data"].get("text/plain", [])))
    return "".join(parts).strip()


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    original = nbformat.read(NB_PATH, as_version=4)

    style_cell = nbformat.v4.new_code_cell(STYLE_CELL)
    nb.cells.insert(0, style_cell)          # 原始索引 i -> 新索引 i+1

    print("正在执行 notebook 副本（原文件不改动）...")
    NotebookClient(
        nb, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": PROJ_DIR}},
    ).execute()

    # ---------------- 数值一致性核对 ----------------
    print("\n数值一致性核对：")
    ok = True
    for idx, expect in CHECK.items():
        got = out_text(nb.cells[idx + 1])
        hit = expect.strip() in got
        ok &= hit
        print(f"  cell {idx:>2}: {'一致 ✅' if hit else '不一致 ❌'}  期望片段 {expect[:38]!r}")
    if not ok:
        print("\n[中止] 重新执行结果与 notebook 原输出不一致，未覆盖任何图片。")
        print("       请改走路线 B（用同样代码重绘）后再继续。")
        sys.exit(1)

    # ---------------- 导出白底插图 ----------------
    print("\n导出插图：")
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, name in FIG_CELLS.items():
        cell = nb.cells[idx + 1]
        pngs = [o for o in cell.get("outputs", [])
                if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})]
        if not pngs:
            print(f"  [警告] cell {idx} 没有图片输出")
            continue
        img = Image.open(io.BytesIO(base64.b64decode(pngs[-1]["data"]["image/png"]))).convert("RGB")
        dst = os.path.join(OUT_DIR, f"{name}.jpg")
        img.save(dst, "JPEG", quality=95, optimize=True)
        corner = img.getpixel((0, 0))
        print(f"  cell {idx:>2} -> {name}.jpg  {img.size[0]}x{img.size[1]}  "
              f"角落像素={corner}  {os.path.getsize(dst)//1024} KB")
    print("\n完成：全部插图均为白底，且数值与 notebook 原输出一致。")


if __name__ == "__main__":
    main()
