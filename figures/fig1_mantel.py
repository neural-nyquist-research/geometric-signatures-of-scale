#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Redraw Figure 1 at the size it is printed.

The submission figure is 4158 x 2375 px, drawn on a large canvas. Printed at a single column
(3.5 in) that canvas scales by about a fifth, and the annotations, legend and tick labels come
out at roughly 3-4 pt -- unreadable, and nothing about the data requires them to be.

This version is drawn at 3.5 x 2.45 in with typography set for that width (ticks 6, axis 6.5,
annotation 5.8, legend 5.5) and carries only the two landmarks the text refers to: the global
maximum and the mid-depth anchor. The four extraction variants stay, thin, because their agreeing
is the point of the panel.

    python replot_fig1_mantel_print.py
    python replot_fig1_mantel_print.py --csv ../paper_revise/supplementary/...  # any cache
"""

import argparse
import collections
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

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CANON = "Norm=True, Metric=Cosine"
COLOURS = {
    "Norm=True, Metric=Cosine": ("#0072B2", "-", 1.9, "cosine, normalised"),
    "Norm=False, Metric=Cosine": ("#56B4E9", "-", 1.1, "cosine"),
    "Norm=True, Metric=Euclidean": ("#D55E00", "--", 1.1, "euclidean, normalised"),
    "Norm=False, Metric=Euclidean": ("#E69F00", ":", 1.2, "euclidean"),
}


def load(path, model="xls-r-2b"):
    curves = collections.defaultdict(dict)
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["Model"].strip().lower() != model:
                continue
            curves[r["Label"].strip()][int(r["Layer"])] = float(r["Mantel_R"])
    if not curves:
        raise SystemExit("[ERROR] no rows for model %r in %s" % (model, path))
    return curves


def draw(curves, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.45), dpi=400)

    for label, (colour, style, lw, name) in COLOURS.items():
        if label not in curves:
            continue
        pts = curves[label]
        L = sorted(pts)
        ax.plot(L, [pts[l] for l in L], style, color=colour, lw=lw, label=name,
                zorder=3 if label == CANON else 2)

    canon = curves[CANON]
    L = sorted(canon)
    y = np.array([canon[l] for l in L])
    top = int(np.argmax(y))
    peak_layer, peak = L[top], y[top]

    # the anchor is a convention, not a feature of the profile: draw it as one
    ax.axvline(25, color="#666666", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.annotate("anchor L25\n%.3f" % canon[25], xy=(25, canon[25]), xytext=(27.5, 0.115),
                fontsize=5.6, color="#444444", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666", shrinkA=0, shrinkB=2))

    ax.annotate("global max L%d\n%.3f" % (peak_layer, peak), xy=(peak_layer, peak),
                xytext=(peak_layer, 0.605), fontsize=5.8, color="#1A1A1A", ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.45, color="#333333", shrinkA=1, shrinkB=2))

    ax.set_xlabel("layer", fontsize=6.5)
    ax.set_ylabel("Mantel $r_M$", fontsize=6.5)
    ax.set_xlim(0, 48)
    ax.set_ylim(0.0, 0.62)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48])
    ax.tick_params(labelsize=6, length=2.0, pad=1.5, width=0.5)
    ax.grid(alpha=0.18, lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)

    ax.legend(fontsize=5.2, frameon=False, loc="lower left", handlelength=1.5,
              handletextpad=0.35, borderaxespad=0.25, labelspacing=0.25)

    fig.tight_layout(pad=0.15)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print("wrote %s  (peak L%d = %.3f, anchor L25 = %.3f)" % (out, peak_layer, peak, canon[25]))
    return peak_layer, peak


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--csv", default=_resolve(os.path.join("re-organized", "mantel_results_cache.csv")))
    ap.add_argument("--out", default=os.path.join(here, "fig1_mantel.png"))
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        print("[ERROR] missing %s" % a.csv)
        return 2
    curves = load(a.csv)
    if CANON not in curves:
        print("[ERROR] canonical variant %r absent" % CANON)
        return 3
    draw(curves, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
