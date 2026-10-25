# -*- coding: utf-8 -*-
"""执行 exp4.ipynb：注入白底高分辨率绘图样式，执行后把输出写回 notebook 本身。"""
import base64, io, os, sys
import nbformat
from nbclient import NotebookClient
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
PROJ_DIR = os.path.dirname(REPORT_DIR)
NB_PATH = os.path.join(PROJ_DIR, "exp4.ipynb")
OUT_DIR = os.path.join(REPORT_DIR, "素材", "exp4", "notebook_out")

STYLE_CELL = '''# === 临时样式注入（仅控制绘图外观，执行后会从 notebook 中移除）===
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
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
    return "".join(parts)

def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(STYLE_CELL))
    print("正在执行 exp4.ipynb ...")
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": PROJ_DIR}}).execute()
    executed = nbformat.v4.new_notebook(metadata=nb.metadata)
    executed.cells = nb.cells[1:]

    print("\n关键输出：")
    for i in (18, 34):
        print(f"\n--- cell {i} ---\n{out_text(executed.cells[i]).strip()[:500]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n导出插图：")
    for i, c in enumerate(executed.cells):
        for j, o in enumerate(c.get("outputs", [])):
            if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}):
                img = Image.open(io.BytesIO(base64.b64decode(o["data"]["image/png"]))).convert("RGB")
                dst = os.path.join(OUT_DIR, f"cell{i:02d}_{j}.jpg")
                img.save(dst, "JPEG", quality=95, optimize=True)
                print(f"  cell {i:>2} -> {os.path.basename(dst)}  {img.size[0]}x{img.size[1]}"
                      f"  角落={img.getpixel((0, 0))}")
    nbformat.write(executed, NB_PATH)
    print("\n已写回:", NB_PATH)

if __name__ == "__main__":
    main()
