"""Draw the two figures used in README.md (no model required).

    python make_figures.py

    assets/vit_architecture.png  -- the whole pipeline + one encoder block, with shapes
    assets/patchify.png          -- image -> 196 patches -> token sequence, on a real photo

Fonts: tries Chinese fonts (Microsoft YaHei etc.) and falls back to English labels
automatically, so the script also runs on a machine without CJK fonts.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from PIL import Image

# --------------------------------------------------------------------------------------
# fonts: pick a CJK font if we have one
# --------------------------------------------------------------------------------------
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "PingFang SC", "DengXian", "SimSun"]
_have = {f.name for f in fm.fontManager.ttflist}
CJK = next((c for c in CJK_CANDIDATES if c in _have), None)
if CJK:
    plt.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def t(zh, en):
    """Pick Chinese or English label depending on font availability."""
    return zh if CJK else en


INK = "#16233a"
SUB = "#41557a"
BLUE, BLUE_FC = "#3b6cb8", "#e9f1fd"
ORANGE, ORANGE_FC = "#c26a12", "#fdf1e2"
GREEN, GREEN_FC = "#2f7d5b", "#e7f4ee"
GREY = "#6b7a90"
PURPLE, PURPLE_FC = "#6b4fa1", "#f0ebfa"


def box(ax, cx, cy, w, h, title, sub=None, fc=BLUE_FC, ec=BLUE, title_fs=11.5, sub_fs=9, zorder=2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                fc=fc, ec=ec, lw=1.5, zorder=zorder))
    if sub:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center", fontsize=title_fs,
                color=INK, weight="bold", zorder=zorder + 1)
        ax.text(cx, cy - h * 0.2, sub, ha="center", va="center", fontsize=sub_fs,
                color=SUB, zorder=zorder + 1, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_fs, color=INK,
                weight="bold", zorder=zorder + 1)


def arrow(ax, p0, p1, color=GREY, lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, lw=lw,
                                 color=color, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def oplus(ax, cx, cy, r=0.55, label="⊕"):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=GREY, lw=1.4, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=13, color=GREY, zorder=5)


# ======================================================================================
# Figure 1: the architecture
# ======================================================================================
def figure_architecture(path):
    fig, ax = plt.subplots(figsize=(14.6, 8.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(2, 97.5, t("Vision Transformer（ViT-B/16）：一次前向传播就是这几步",
                       "Vision Transformer (ViT-B/16): one forward pass"),
            fontsize=15, weight="bold", color=INK)

    # ---------------- left column: the pipeline ----------------
    x = 17
    stages = [
        (89, t("输入图像", "input image"), "x ∈ R^(1×3×224×224)", BLUE_FC, BLUE),
        (78, t("切成 14×14 个 16×16 patch", "split into 14×14 patches of 16×16"),
         t("196 个 patch，每个 16·16·3 = 768 维", "196 patches, each 16·16·3 = 768 dims"), BLUE_FC, BLUE),
        (67, t("线性投影 E  （即 Eq.1）", "linear projection E  (Eq.1)"),
         t("768 → 768，token 序列 196 × 768", "768 → 768, tokens 196 × 768"), BLUE_FC, BLUE),
        (56, t("前面拼 [CLS]，再加位置编码", "prepend [CLS], add pos-embed"),
         t("197 × 768（1 个 CLS + 196 个 patch）", "197 × 768 (1 CLS + 196 patches)"), BLUE_FC, BLUE),
    ]
    for y, title, sub, fc, ec in stages:
        box(ax, x, y, 26, 8.2, title, sub, fc=fc, ec=ec)
    labels = [t("patchify", "patchify"), t("E ∈ R^(768×768)", "E ∈ R^(768×768)"),
              t("+ [CLS] + pos", "+ [CLS] + pos")]
    for (y0, *_), y1, lab in zip(stages[:-1], [s[0] for s in stages[1:]], labels):
        arrow(ax, (x, y0 - 4.1), (x, y1 + 4.1))
        ax.text(x + 0.8, (y0 + y1) / 2, lab, fontsize=8.5, color=GREY, va="center")

    box(ax, x, 42, 26, 10, t("Transformer Encoder  × 12", "Transformer Encoder  × 12"),
        t("每层形状不变：197 × 768", "shape-preserving: 197 × 768"), fc=ORANGE_FC, ec=ORANGE, title_fs=12.5)
    arrow(ax, (x, 51.9), (x, 47.1))
    arrow(ax, (x + 13.2, 42), (49, 42), color=ORANGE, lw=1.8, rad=0.0)
    ax.text(41, 43.6, t("放大 →", "zoom →"), fontsize=9, color=ORANGE)

    arrow(ax, (x, 36.9), (x, 32.1))
    ax.text(x + 0.8, 34.6, "LayerNorm", fontsize=8.5, color=GREY, va="center")

    box(ax, x, 28, 26, 7.6, t("只取 [CLS] 那一个 token", "take the [CLS] token"),
        t("197 × 768  →  768", "197 × 768  →  768"), fc=GREEN_FC, ec=GREEN)
    arrow(ax, (x, 24.1), (x, 19.9))
    box(ax, x, 16, 26, 7.4, t("MLP Head（768 → 1000）", "MLP head (768 → 1000)"),
        t("= 一个 Linear + softmax", "one Linear + softmax"), fc=GREEN_FC, ec=GREEN)
    arrow(ax, (x, 12.2), (x, 9.4))
    box(ax, x, 6, 26, 5.6, t("1000 类 logits", "1000 logits"), fc="white", ec=GREY)

    # footnotes
    ax.text(2, 1.6, t("总计 86.6M 参数 · 17.7 GMAC / 图 · 没有一处卷积（除 patch 切分本身）",
                     "86.6M params · 17.7 GMAC/image · zero convolutions except the patch split"),
            fontsize=9, color=GREY)

    # ---------------- right panel: one encoder block ----------------
    px0, px1 = 47, 99
    ax.add_patch(FancyBboxPatch((px0, 12), px1 - px0, 82, boxstyle="round,pad=0.02,rounding_size=1.2",
                                fc="#fbfcfe", ec="#c9d4e4", lw=1.2, zorder=0))
    ax.text((px0 + px1) / 2, 90, t("一个 Encoder Block 内部（12 层完全相同）", "inside one encoder block (x12)"),
            fontsize=12, weight="bold", color=INK, ha="center")

    xin, xatt = 60, 0
    # x inlet
    ax.text(px0 + 2.2, 81.5, "x", fontsize=12, weight="bold", color=INK)
    ax.text(px0 + 3.6, 81.5, t("(197 × 768)", "(197 × 768)"), fontsize=9, color=SUB)
    arrow(ax, (px0 + 2, 80.4), (px0 + 2, 76.5), color=INK, lw=1.4)

    # branch: top path (attention) and bottom path (MLP) from the same x
    ax.plot([px0 + 2, px0 + 2], [83.2, 20], color=INK, lw=1.4, zorder=1)          # trunk
    ax.plot([px0 + 2, 55.4], [76.5, 76.5], color=INK, lw=1.4, zorder=1)
    ax.plot([px0 + 2, 55.4], [52, 52], color=INK, lw=1.4, zorder=1)

    box(ax, 58.4, 76.5, 6.2, 6.4, t("LN", "LN"), fc="white", ec=GREY, title_fs=10.5)
    arrow(ax, (61.5, 76.5), (64.4, 76.5), color=INK)
    box(ax, 76, 76.5, 22, 10.4, t("多头自注意力 MSA", "multi-head self-attention"),
        t("h = 12 头, d_h = 64\nA = softmax(q·k^T / √64)：197 × 197\nout = A·v·W_O：197 × 768",
          "h = 12 heads, d_h = 64\nA = softmax(q·k^T / √64): 197 × 197\nout = A·v·W_O: 197 × 768"),
        fc=ORANGE_FC, ec=ORANGE, title_fs=11)
    arrow(ax, (87, 76.5), (90.4, 76.5), color=INK)
    oplus(ax, 91.6, 76.5)
    arrow(ax, (92.2, 76.5), (96.5, 76.5), color=INK)

    # residual arc into the first oplus
    ax.annotate("", xy=(91.6, 77.1), xytext=(91.6, 83.2),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                                connectionstyle="arc3,rad=0.0", shrinkA=0, shrinkB=0))
    ax.plot([px0 + 2, 91.6], [83.2, 83.2], color=GREEN, lw=1.5, ls=(0, (5, 3)), zorder=1)
    ax.plot([91.6, 91.6], [83.2, 77.1], color=GREEN, lw=1.5, ls=(0, (5, 3)), zorder=1)
    ax.text(83, 84.4, t("残差直连（identity）", "residual"), fontsize=8.5, color=GREEN)

    # second sub-block
    ax.text(96.6, 79.4, "x'", fontsize=11, weight="bold", color=INK)
    ax.plot([96.5, 96.5], [76.5, 52], color=INK, lw=1.4, zorder=1)
    ax.plot([96.5, 96.5], [52, 48], color=INK, lw=1.4, zorder=1)
    box(ax, 58.4, 52, 6.2, 6.4, t("LN", "LN"), fc="white", ec=GREY, title_fs=10.5)
    ax.plot([96.5, 55.4], [52, 52], color=INK, lw=1.4, zorder=1)
    arrow(ax, (61.5, 52), (64.4, 52), color=INK)
    box(ax, 76, 52, 22, 10.4, t("MLP（逐 token 独立）", "MLP (per token, independent)"),
        t("768 → 3072 → 768, GELU\n占单层参数的 67%",
          "768 → 3072 → 768, GELU\n67% of a block's params"),
        fc=PURPLE_FC, ec=PURPLE, title_fs=11)
    arrow(ax, (87, 52), (90.4, 52), color=INK)
    oplus(ax, 91.6, 52)
    arrow(ax, (92.2, 52), (96.5, 52), color=INK)
    ax.plot([px0 + 2, 91.6], [45, 45], color=GREEN, lw=1.5, ls=(0, (5, 3)), zorder=1)
    ax.plot([91.6, 91.6], [45, 52.6], color=GREEN, lw=1.5, ls=(0, (5, 3)), zorder=1)
    ax.annotate("", xy=(91.6, 51.4), xytext=(91.6, 45),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, shrinkA=0, shrinkB=0))
    ax.plot([px0 + 2, px0 + 2], [45, 20], color=INK, lw=1.4, zorder=1)

    ax.text(px0 + 2.2, 18.5, t("输出 (197 × 768)，送进下一层", "output (197 x 768) -> next block"),
            fontsize=9.5, color=INK)

    # what attention actually does
    ax.add_patch(FancyBboxPatch((px0 + 1.5, 22.5), 49, 19.5, boxstyle="round,pad=0.02,rounding_size=0.9",
                                fc="white", ec="#dde4ee", lw=1.0, zorder=2))
    ax.text(px0 + 3, 39.8, t("“每个 token 看所有 token”是什么意思", "'every token looks at all tokens'"),
            fontsize=10, weight="bold", color=INK)
    lines = [
        t("① 197 个 token 各自乘 3 个矩阵，得到 q、k、v（197 × 768）",
          "1) each of the 197 tokens makes q, k, v (197 x 768)"),
        t("② q 和 k 两两点积 → 197 × 197 的分数矩阵：第 i 行 = “token i 和谁相关”",
          "2) pairwise q·k -> 197 x 197 scores; row i = who token i cares about"),
        t("③ softmax 把每一行归一化成权重（行和 = 1）",
          "3) softmax normalises each row (sums to 1)"),
        t("④ 用权重对 v 加权求和 → 每个 token 变成“大家信息的混合”",
          "4) weighted sum of v -> each token becomes a mix of all tokens"),
        t("⑤ 全流程的“图像空间结构”只来自 pos-embed，别的都不知道 2D 位置",
          "5) all 2D structure comes from pos-embed; nothing else knows position"),
    ]
    for i, ln in enumerate(lines):
        ax.text(px0 + 3, 36.2 - i * 2.9, ln, fontsize=8.8, color=SUB)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ======================================================================================
# Figure 2: patchify on a real photo
# ======================================================================================
def figure_patchify(path, img_path="assets/sample.jpg", patch=16):
    """(a) the image cut into a grid  (b) one row of patches laid out as a token sequence."""
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    a = np.asarray(img)
    g = 224 // patch  # 14

    fig = plt.figure(figsize=(13.2, 4.9))

    # ---------------- (a) image + grid ----------------
    ax0 = fig.add_axes([0.01, 0.06, 0.30, 0.84])
    ax0.imshow(a)
    for i in range(g + 1):
        ax0.axhline(i * patch - 0.5, color="white", lw=0.6, alpha=0.75)
        ax0.axvline(i * patch - 0.5, color="white", lw=0.6, alpha=0.75)
    for c in range(g):  # row-major order made explicit on the first row
        ax0.text(c * patch + patch / 2, patch / 2, str(c), color="white", fontsize=7,
                 ha="center", va="center", weight="bold",
                 bbox=dict(fc="black", alpha=0.5, pad=1.0, ec="none"))
    for idx in (g, g * g - 1):
        r, c = divmod(idx, g)
        ax0.add_patch(plt.Rectangle((c * patch - 0.5, r * patch - 0.5), patch, patch,
                                    fill=False, ec="#ffd166", lw=2.2))
        ax0.text(c * patch + patch / 2, r * patch + patch / 2, str(idx), color="white",
                 fontsize=7.5, ha="center", va="center", weight="bold",
                 bbox=dict(fc="black", alpha=0.6, pad=1.2, ec="none"))
    ax0.axis("off")
    fig.text(0.16, 0.965, t(f"(a) 224×224 切成 14×14 = {g*g} 个 16×16 patch",
                           f"(a) 224x224 cut into {g*g} patches of 16x16"),
             fontsize=10.5, ha="center", weight="bold", color=INK)
    fig.text(0.16, 0.015, t("编号 = 行优先顺序（先读第 0 行，再第 1 行…）",
                            "indices are row-major (read row 0, then row 1, ...)"),
             fontsize=8, ha="center", color=GREY)

    # ---------------- (b) a row of patches as a token sequence ----------------
    ax1 = fig.add_axes([0.35, 0.08, 0.64, 0.84])
    ax1.set_xlim(-2.3, 16.2); ax1.set_ylim(0, 10); ax1.axis("off")

    ax1.add_patch(plt.Rectangle((-1.95, 4.2), 1.0, 2.6, fc="#2f7d5b", ec="none"))
    ax1.text(-1.45, 5.5, "[CLS]", color="white", fontsize=7.5, ha="center", va="center",
             weight="bold", rotation=90)
    for c in range(g):
        ax1.imshow(a[0:patch, c * patch:(c + 1) * patch],
                   extent=(c - 0.45, c + 0.45, 4.4, 7.4), zorder=3)
    ax1.text(14.6, 5.9, "···", fontsize=15, color=SUB, ha="center", va="center")

    ax1.plot([-1.95, 15.4], [8.4, 8.4], color=GREY, lw=1.0)
    ax1.text(6.7, 9.15, t("位置 0 是 [CLS]，位置 1…196 是 patch token：序列长度 197",
                          "position 0 is [CLS], positions 1..196 are patch tokens: length 197"),
             fontsize=9, color=SUB, ha="center")
    ax1.text(-1.45, 5.5, "", fontsize=8)
    ax1.text(-1.45, 8.75, "0", fontsize=8, color="#2f7d5b", ha="center")
    for c in range(g):
        ax1.text(c, 8.75, str(c + 1), fontsize=8, color=SUB, ha="center")
        ax1.text(c, 3.5, str(c + 1), fontsize=7.5, color=SUB, ha="center")
    ax1.text(-1.45, 3.5, "0", fontsize=7.5, color="#2f7d5b", ha="center")

    ax1.text(6.7, 1.6, t("每个格子 = 16×16×3 = 768 个像素值，乘上 E (768×768) 变成 1 个 768 维 token",
                         "each cell = 16x16x3 = 768 pixel values, multiplied by E (768x768) into one 768-d token"),
             fontsize=8.5, color=GREY, ha="center")
    ax1.text(6.7, 0.4, t("第 0 行 → 第 1 行 → …… → 第 13 行（行优先，和读文字一样）",
                         "row 0 -> row 1 -> ... -> row 13 (row-major, like reading text)"),
             fontsize=8.5, color=GREY, ha="center")

    fig.text(0.67, 0.965, t("(b) 摊平成模型真正读到的“一句话”", "(b) flattened into the sentence the model reads"),
             fontsize=10.5, ha="center", weight="bold", color=INK)

    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    figure_architecture("assets/vit_architecture.png")
    figure_patchify("assets/patchify.png")
