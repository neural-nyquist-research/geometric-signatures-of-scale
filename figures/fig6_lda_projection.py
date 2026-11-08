#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Redraw Fig. 6 (tone-discriminant axis at L38) in the style of Fig. 5.

Why this exists
---------------
The figure in the paper came from `08_lda_tone_analysis_Fig07_..._Fig08_fig-lda_projection.py`,
which draws a 10 x 2 inch strip with all 25 language labels rotated 45 degrees. At the
printed width (7.16 in) the labels collide, the colours are pure blue/red (not
colour-blind safe) and the type is larger than in every other figure. The projection
values themselves are correct; only the rendering was wrong.

`08` now dumps the projection to `lda_projection_L38.csv` instead of plotting. This
script reads that CSV and draws the figure with the same typography as
`plot_id_trajectories_4models.py` (Fig. 5) and the Okabe-Ito palette.

Palette
-------
Okabe & Ito (2008): #0072B2 blue and #D55E00 vermillion. Chosen because they stay
distinguishable under deuteranopia, protanopia and tritanopia and also differ in
luminance, so the two classes remain separable in greyscale print.

Label layout
------------
Labels alternate above and below the axis and are pushed apart horizontally when
they would collide, so nothing overlaps at print size. Rucheng is set in bold
because it is the stress case the text discusses.

Usage
-----
    python replot_fig6_from_csv.py                      # reads lda_projection_L38.csv
    python replot_fig6_from_csv.py --csv other.csv --out other.png
"""

import argparse
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

TOL = "#D55E00"      # tonal  (Okabe-Ito vermillion)
NON = "#0072B2"      # non-tonal (Okabe-Ito blue)


def read_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        lang = (r.get("language") or r.get("Language") or "").strip()
        val = (r.get("lda1") or r.get("LDA1") or r.get("value") or "").strip()
        tone = (r.get("tone") or r.get("Tone") or "").strip()
        if not lang or not val:
            continue
        out.append((lang, float(val), int(float(tone)) if tone else 0))
    return out


def place_labels(ax, pts):
    """Greedy interval packing: alternate sides, then the first free tier."""
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    NTIER = 4
    last = {1: [None] * NTIER, -1: [None] * NTIER}
    for i, (lang, x, _tone) in enumerate(sorted(pts, key=lambda p: p[1])):
        half = 0.0040 * len(lang) * span          # ~3.3 pt per character
        side = 1 if i % 2 == 0 else -1
        tier = NTIER - 1
        for t in range(NTIER):
            if last[side][t] is None or (x - half) > last[side][t] + 0.006 * span:
                tier = t
                break
        last[side][tier] = x + half
        y = side * (0.14 + 0.15 * tier)
        ax.annotate(lang, (x, 0.0), xytext=(x, y), textcoords="data",
                    ha="center", va="center", fontsize=5.9, color="#222222",
                    fontweight="bold" if lang.lower() == "rucheng" else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.35, color="#8A8A8A",
                                    shrinkA=0, shrinkB=1.5))


def draw(pts, out_path):
    fig, ax = plt.subplots(figsize=(7.16, 2.55), dpi=300)

    for lang, x, tone in pts:
        ax.scatter([x], [0.0], s=26, marker="o",
                   facecolor=TOL if tone else NON, edgecolor="white",
                   linewidth=0.6, zorder=5)

    ax.axvline(0.0, color="#666666", linestyle=(0, (3, 2)), linewidth=0.7, alpha=0.55, zorder=1)
    ax.axhline(0.0, color="#BBBBBB", linewidth=0.5, zorder=0)

    ax.set_ylim(-0.86, 0.86)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", labelsize=7, length=2.5, pad=2, colors="#333333")
    ax.grid(axis="x", alpha=0.20, linewidth=0.5)

    ax.set_xlabel("Tone-discriminant axis (LDA-1, Layer 38)", fontsize=8)
    ax.set_title("Tone-discriminant axis at Layer 38", fontsize=9.0, pad=6)

    # legend, matching the marker style
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.2,
                    markerfacecolor=NON, markeredgecolor="white", markeredgewidth=0.5,
                    label="non-tonal"),
         plt.Line2D([], [], marker="o", linestyle="", markersize=4.2,
                    markerfacecolor=TOL, markeredgecolor="white", markeredgewidth=0.5,
                    label="tonal")]
    ax.legend(handles=h, loc="upper left", fontsize=6.6, frameon=False,
              handletextpad=0.4, borderaxespad=0.3)

    ax.margins(x=0.06)
    place_labels(ax, pts)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote %s" % out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=_resolve(os.path.join("data", "lda_projection_L38.csv")))
    ap.add_argument("--out", default=os.path.join(here, "fig6_lda_projection.png"))
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        print("[ERROR] missing %s" % a.csv)
        print("        Run 08_lda_tone_analysis_Fig07_..._Fig08_fig-lda_projection.py "
              "on the machine that holds hidden_states_38/; it now writes this CSV.")
        return 2
    pts = read_csv(a.csv)
    if len(pts) < 20:
        print("[ERROR] only %d languages in %s -- expected 25" % (len(pts), a.csv))
        return 3
    draw(pts, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
