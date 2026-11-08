#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-plot the intrinsic-dimension trajectory figure with all FOUR XLS-R models.

The original PNG (ID_Language_Paths_Comparison_Normalized_Annotated.png) showed
XLS-R-2B against XLSR-53 only, at 4156x2368 px (13.9x7.9 in) inside a single-column
figure -- a 0.25x reduction that left ~3 pt type. It also predates the 1B model,
which is the whole point of the family comparison: without it a two-point difference
cannot be distinguished from a step.

This script redraws it at final print size (7.16 in, full text width) with
  * mean +/- 1 s.d. across the 24 FLEURS languages, per model
  * the Rucheng dialect as a thin dotted line, per model
  * a star at each model's mid-depth anchor (a convention, not a Mantel selection)

Data: re-organized/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv
      (rows ending in _fleurs are the 24 shared languages; _hn_rucheng is the dialect;
       GLOBAL_MEAN is skipped)
"""
import csv
import os

def _resolve(rel):
    """Find a data path whether the script sits in figures/ or in a flat unzip."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here)):
        p = os.path.abspath(os.path.join(base, rel))
        if os.path.exists(p):
            return p
    return os.path.abspath(os.path.join(here, rel))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = _resolve("re-organized")
OUT = os.path.join(HERE, "fig5_id_language_paths.png")

MODELS = [
    ("XLSR-53",    "wav2vec2-large-xlsr-53", 13, "#E69F00"),
    ("XLS-R-300M", "wav2vec2-xls-r-300m",    13, "#56B4E9"),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b",      25, "#009E73"),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b",      25, "#0072B2"),
]
RUCHENG = "_hn_rucheng"


def load_model(model_dir):
    """Return (layers, rel_depth, mean, sd, rucheng) over all available layers."""
    idir = os.path.join(ROOT, model_dir, "ID_result")
    if not os.path.isdir(idir):
        return None
    layers = sorted(int(f.split("_L")[1].split("_")[0])
                    for f in os.listdir(idir)
                    if f.startswith("ID_Analysis_L") and f.endswith("_Fixed.csv"))
    mean, sd, ruc = [], [], []
    for L in layers:
        vals, r = [], np.nan
        with open(os.path.join(idir, f"ID_Analysis_L{L}_Fixed.csv"), newline="",
                  encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lang = row["Language"]
                if lang.endswith("_fleurs"):
                    vals.append(float(row["Intrinsic_Dim"]))
                elif lang == RUCHENG:
                    r = float(row["Intrinsic_Dim"])
        mean.append(np.mean(vals) if vals else np.nan)
        sd.append(np.std(vals, ddof=1) if len(vals) > 1 else np.nan)
        ruc.append(r)
    layers = np.asarray(layers)
    return layers, layers / layers.max(), np.asarray(mean), np.asarray(sd), np.asarray(ruc)


def main():
    fig, ax = plt.subplots(figsize=(7.16, 3.45), dpi=300)

    for name, mdir, L_bt, colour in MODELS:
        got = load_model(mdir)
        if got is None:
            print("[WARN] missing ID_result for", mdir)
            continue
        layers, rel, mean, sd, ruc = got
        ax.fill_between(rel, mean - sd, mean + sd, color=colour, alpha=0.16, linewidth=0)
        ax.plot(rel, mean, "-", color=colour, linewidth=1.5,
                label=f"{name} (mean $\\pm$1 s.d.)")
        ax.plot(rel, ruc, ":", color=colour, linewidth=0.9, alpha=0.85)

        k = int(np.where(layers == L_bt)[0][0])
        ax.plot(rel[k], mean[k], "*", color=colour, markersize=9,
                markeredgecolor="white", markeredgewidth=0.5, zorder=6)

    # 半透明带只标出"约定锚点所在的位置"，不是功能阶段：
    # 论文 §IV-A 明确 mid-depth 层是 round(0.5(n-1))+1 的约定，Mantel 剖面并不选它。
    ax.axvspan(0.50, 0.56, color="grey", alpha=0.10)
    ax.text(0.53, 0.97, "mid-depth\nanchor", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.8, color="#444444")

    ax.set_xlabel("Relative depth  (0 = input, 1 = output)", fontsize=8)
    ax.set_ylabel("Intrinsic dimension  $\\widehat{id}$", fontsize=8)
    ax.set_title("Intrinsic dimension across matched relative depth", fontsize=9.0, pad=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(7.5, 20.0)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.tick_params(labelsize=7, length=2.5, pad=1.8)
    ax.legend(fontsize=6.4, loc="lower left", framealpha=0.92, borderpad=0.35,
              handlelength=2.0, labelspacing=0.3)

    # 说明文字只保留在 LaTeX 图注里，避免图内与图注重复（IEEE 惯例）
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
