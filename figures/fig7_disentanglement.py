#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Redraw the tone/syllable disentanglement scatter at print size.

The submission's version of this figure is 8 x 8 in at low dpi. Square, so at any
font size that is legible in print it occupies 5-7 in of page height, and reduced to
3.5 in its labels fall to about 4 pt. The data, not the canvas, is what the paper
needs, so `07_lda_tone_vs_syllstr_analysis_...py` now dumps the two per-dimension
weight vectors and this script draws them wide.

Layout, in the same typography as the other figures (7.16 in, 300 dpi, title 9,
axis 8, ticks 7), with the Okabe-Ito palette:
  - all 1920 dimensions in grey,
  - the top 10 dimensions of each family in vermillion (tone) and blue (syllable),
    every one of them annotated with its index, placed by a greedy anti-overlap
    routine so that twenty numbers stay legible in a 7.16 x 3.05 in panel,
  - the overlap of the top-10 and of the top-20 sets, recomputed here and printed in
    the axes, so the numbers in the caption are not taken on trust. The two are
    reported separately because the panel shows ten per family while the caption's
    claim is about twenty.

Usage
-----
    python replot_fig7_disentanglement.py
    python replot_fig7_disentanglement.py --csv lda_weights_L38.csv --out fig7.png
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
import numpy as np

TOL = "#D55E00"
SYL = "#0072B2"
GREY = "#9A9A9A"

# Candidate label positions in points: angle (deg, 0 = right, CCW) and radius.
ANGLES = (0, 180, 90, 270, 35, 145, 215, 325, 60, 120, 240, 300)
RADII = (7.0, 11.0, 15.5, 21.0, 27.0)


def load(path):
    d, t, s = [], [], []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d.append(int(r["dimension"]))
            t.append(float(r["tone_weight"]))
            s.append(float(r["syllable_weight"]))
    return np.array(d), np.array(t), np.array(s)


def top_overlap(t, s, n):
    a = set(np.argsort(t)[-n:][::-1].tolist())
    b = set(np.argsort(s)[-n:][::-1].tolist())
    inter = a & b
    return inter, len(inter) / len(a | b)


def _overlap_area(b1, b2, pad=1.0):
    dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0) + pad
    dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0) + pad
    return dx * dy if dx > 0 and dy > 0 else 0.0


def place_labels(ax, fig, entries, fontsize):
    """Greedy anti-overlap placement for the annotation set.

    `entries` is a list of (idx, x, y, colour). Each label tries a ring of candidate
    offsets, nearest first, and takes the first position that collides with neither a
    placed label nor one of the highlighted points. Two refinements matter when
    several dimensions of one family sit on almost the same coordinates (1257 and 1891
    differ by less than a marker width): a label whose anchor is near an already
    labelled anchor is pushed to the opposite side, and the collision test keeps the
    twenty leader lines short enough to read.
    """
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer)
    placed = []          # (bbox, anchor display coords, dx sign)
    pts = []
    for _, x, y, _ in entries:
        px, py = ax.transData.transform((x, y))
        pts.append(matplotlib.transforms.Bbox.from_bounds(px - 2.8, py - 2.8, 5.6, 5.6))

    order = sorted(entries, key=lambda e: -max(e[1], e[2]))
    for idx, x, y, colour in order:
        px, py = ax.transData.transform((x, y))
        best = None          # (cost, dx, dy, ha, va)
        for radius in RADII:
            for ang in ANGLES:
                a = np.deg2rad(ang)
                dx, dy = radius * np.cos(a), radius * np.sin(a)
                ha = "left" if np.cos(a) > 0.25 else ("right" if np.cos(a) < -0.25 else "center")
                va = "bottom" if np.sin(a) > 0.25 else ("top" if np.sin(a) < -0.25 else "center")
                probe = ax.annotate("%d" % idx, (x, y), xytext=(dx, dy),
                                    textcoords="offset points", fontsize=fontsize,
                                    ha=ha, va=va, color=colour)
                bb = probe.get_window_extent(renderer)
                probe.remove()
                if (bb.x0 < ax_bb.x0 + 1 or bb.x1 > ax_bb.x1 - 1
                        or bb.y0 < ax_bb.y0 + 1 or bb.y1 > ax_bb.y1 - 1):
                    continue
                cost = sum(_overlap_area(bb, p[0]) for p in placed)
                cost += 3.0 * sum(_overlap_area(bb, p) for p in pts)
                # Do not stack two labels on the same side of nearly coincident points.
                for _, (qx, qy), qsign in placed:
                    if abs(px - qx) < 10 and abs(py - qy) < 10:
                        sign = 1 if dx > 0.5 else (-1 if dx < -0.5 else 0)
                        if sign != 0 and sign == qsign:
                            cost += 4.0e4
                if best is None or cost < best[0]:
                    best = (cost, dx, dy, ha, va)
                if cost == 0:
                    break
            if best is not None and best[0] == 0:
                break
        if best is None:
            best = (1e18, 7.0, 0.0, "left", "center")
        _, dx, dy, ha, va = best
        ann = ax.annotate("%d" % idx, (x, y), xytext=(dx, dy), textcoords="offset points",
                          fontsize=fontsize, color=colour, ha=ha, va=va,
                          arrowprops=dict(arrowstyle="-", lw=0.55, color=colour,
                                          alpha=0.85, shrinkA=0, shrinkB=1.2))
        placed.append((ann.get_window_extent(renderer), (px, py),
                       1 if dx > 0.5 else (-1 if dx < -0.5 else 0)))


def draw(d, t, s, out, top_n=10, fontsize=6.2):
    inter10, jac10 = top_overlap(t, s, top_n)
    inter20, jac20 = top_overlap(t, s, 20)
    top_t = np.argsort(t)[-top_n:][::-1]
    top_s = np.argsort(s)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(7.16, 3.05), dpi=300)
    fig.canvas.draw()

    ax.scatter(np.maximum(t, 0), np.maximum(s, 0), s=6, c=GREY, alpha=0.45,
               linewidths=0, zorder=1, rasterized=True)
    ax.scatter(t[top_s], s[top_s], s=22, facecolor=SYL, edgecolor="white",
               linewidth=0.5, zorder=3, label="top %d syllable importance" % top_n)
    ax.scatter(t[top_t], s[top_t], s=22, facecolor=TOL, edgecolor="white",
               linewidth=0.5, zorder=3, label="top %d tone importance" % top_n)

    ax.margins(x=0.14, y=0.20)
    ax.set_xlabel("Tone LDA importance", fontsize=8)
    ax.set_ylabel("Syllable LDA importance", fontsize=8)
    ax.set_title("Tone and syllable importance at Layer 38, XLS-R-2B", fontsize=9, pad=5)
    ax.tick_params(labelsize=7, length=2.5, pad=2)
    ax.grid(alpha=0.20, linewidth=0.5, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    entries = [(int(d[i]), float(t[i]), float(s[i]), SYL) for i in top_s]
    entries += [(int(d[i]), float(t[i]), float(s[i]), TOL) for i in top_t]
    place_labels(ax, fig, entries, fontsize)

    # Two sets, two numbers: the panel shows ten per family, the caption claims twenty.
    ax.text(0.985, 0.975,
            "top-%d sets (shown): %d shared\n"
            "top-20 sets: %d shared, Jaccard %.2f%%"
            % (top_n, len(inter10), len(inter20), 100 * jac20),
            transform=ax.transAxes, va="top", ha="right", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC",
                      linewidth=0.5, alpha=0.9))
    ax.legend(fontsize=6.6, frameon=False, loc="lower right", handletextpad=0.3,
              borderaxespad=0.4, markerscale=1.2)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote %s  (top-%d overlap %d, top-20 overlap %d, Jaccard %.4f)"
          % (out, top_n, len(inter10), len(inter20), jac20))
    return inter10, inter20


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=_resolve(os.path.join("data", "lda_weights_L38.csv")))
    ap.add_argument("--out", default=os.path.join(here, "fig7_disentanglement.png"))
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        print("[ERROR] missing %s" % a.csv)
        print("        Run 07_lda_tone_vs_syllstr_...py on the machine that holds "
              "hidden_states_38/; it now writes this CSV.")
        return 2
    d, t, s = load(a.csv)
    if len(d) < 1000:
        print("[ERROR] only %d dimensions -- expected 1920" % len(d))
        return 3
    inter10, inter20 = draw(d, t, s, a.out)
    if len(inter20) != 0:
        print("[NOTE] the top-20 sets overlap in %d dimensions; the caption's "
              "'zero intersection' would need rewording" % len(inter20))
    if len(inter10) != 0:
        print("[NOTE] the top-10 sets overlap in %d dimensions." % len(inter10))
    return 0


if __name__ == "__main__":
    sys.exit(main())
