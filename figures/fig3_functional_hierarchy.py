#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functional-hierarchy figure, four XLS-R models, aligned with the original
`06_rebuttal_problem_04_..._fig-dual_axis_global.py` definitions.

Why this script exists
----------------------
The original plotting script defines the two expertise curves as

    Tone_Expertise    = mean over all rows of |Corr_Height|
                        (global_results/Layer_Evolution_Analysis_allLayers.csv)
    Formant_Expertise = mean over all rows of |Max_Corr|
                        (formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv)

with no significance threshold, and Mean_ID as the plain mean of Intrinsic_Dim per
layer. This script reproduces exactly those definitions, so the figure is traceable
to the same numbers as the original, and re-lays it out at print size for a
two-column journal page.

Deviations from the original, all deliberate:
  * GLOBAL_MEAN rows are excluded from the ID average (the original averaged them
    in, which double-counts one pooled row).
  * 2x2 layout at 7.16 in instead of 1x4 at 17 in, so the type is readable when the
    figure is placed at \\linewidth.
  * Neutral panel titles (the original used "Congested Processing" /
    "Sequential Specialization", which the revision removes).
  * The original emitted three panels in the saved PNG while the script defines
    four; this version plots all four.

Output: fig/Functional_Hierarchy_Segmental_vs_Suprasegmental_4models.png
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
OUT = os.path.join(HERE, "fig3_functional_hierarchy.png")

CONFIGS = [
    ("XLSR-53",    "wav2vec2-large-xlsr-53", 13),
    ("XLS-R-300M", "wav2vec2-xls-r-300m",    13),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b",      25),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b",      25),
]


def load_model(model_dir):
    base = os.path.join(ROOT, model_dir)

    # --- mean intrinsic dimension per layer (GLOBAL_MEAN excluded) ---
    idir = os.path.join(base, "ID_result")
    id_mean = {}
    for f in sorted(os.listdir(idir)):
        if not (f.startswith("ID_Analysis_L") and f.endswith("_Fixed.csv")):
            continue
        layer = int(f.split("_L")[1].split("_")[0])
        vals = []
        with open(os.path.join(idir, f), newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Language"] == "GLOBAL_MEAN":
                    continue
                try:
                    vals.append(float(row["Intrinsic_Dim"]))
                except ValueError:
                    pass
        if vals:
            id_mean[layer] = sum(vals) / len(vals)

    # --- tone expertise: mean |Corr_Height| ---
    tone = {}
    gpath = os.path.join(base, "global_results", "Layer_Evolution_Analysis_allLayers.csv")
    with open(gpath, newline="", encoding="utf-8") as fh:
        acc = {}
        for row in csv.DictReader(fh):
            if row["Language"] == "GLOBAL_MEAN":
                continue
            try:
                v = abs(float(row["Corr_Height"]))
            except ValueError:
                continue
            acc.setdefault(int(row["Layer"]), []).append(v)
    tone = {L: sum(v) / len(v) for L, v in acc.items()}

    # --- formant expertise: mean |Max_Corr| over F1--F3, no threshold ---
    fpath = os.path.join(base, "formant_dim_analysis_cache",
                         "Formant_Lag_Analysis_Results_moreLayers.csv")
    acc = {}
    with open(fpath, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                v = abs(float(row["Max_Corr"]))
            except ValueError:
                continue
            acc.setdefault(int(row["Layer"]), []).append(v)
    formant = {L: sum(v) / len(v) for L, v in acc.items()}

    layers = sorted(set(id_mean) & set(tone) & set(formant))
    x = np.array(layers) / max(layers)
    return (np.array(layers), x,
            np.array([id_mean[L] for L in layers]),
            np.array([formant[L] for L in layers]),
            np.array([tone[L] for L in layers]))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.1), dpi=300, sharex=True)

    for ax, (name, mdir, L_bt) in zip(axes.ravel(), CONFIGS):
        layers, x, idm, formant, tone = load_model(mdir)
        ax2 = ax.twinx()

        ax.plot(x, idm, "-", color="#7f7f7f", linewidth=2.4, alpha=0.55, zorder=1)
        ax2.plot(x, formant, "-", color="#1f77b4", linewidth=1.4, zorder=3)
        ax2.plot(x, tone, "-", color="#d62728", linewidth=1.4, zorder=3)

        ax.set_title(name, fontsize=7.6, pad=3)
        ax.tick_params(labelsize=6.2, length=2, pad=1.5)
        ax2.tick_params(labelsize=6.2, length=2, pad=1.5)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.axvline(L_bt / max(layers), color="#444444", linewidth=0.7,
                   linestyle=":", zorder=2)

    for ax in axes[:, 0]:
        ax.set_ylabel("Intrinsic dim.  $\\widehat{id}$", fontsize=7)
    for ax in axes[:, 1]:
        ax.set_ylabel("mean $|r|$", fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("Relative depth", fontsize=7)

    handles = [plt.Line2D([], [], color="#7f7f7f", linewidth=2.4, alpha=0.55),
               plt.Line2D([], [], color="#1f77b4", linewidth=1.4),
               plt.Line2D([], [], color="#d62728", linewidth=1.4)]
    labels = ["Intrinsic dim. (left)",
              "Formant expertise, mean $|r|$ vs F1--F3 (right)",
              "Tone expertise, mean $|r|$ vs $F_0$ height (right)"]
    fig.legend(handles, labels, fontsize=6.0, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.975), handlelength=2.0,
               columnspacing=1.4)

    fig.suptitle("Segmental and suprasegmental expertise across depth, four XLS-R models",
                 fontsize=8.4, y=1.0)
    fig.text(0.5, -0.03,
             "Dotted vertical line: each model's mid-depth anchor (L13/L13/L25/L25), a convention for comparability. "
             "Curves follow the definitions of the original analysis script: no significance threshold.",
             ha="center", fontsize=5.8, color="#333333")
    fig.tight_layout(rect=(0, 0.01, 1, 0.94), h_pad=1.0, w_pad=1.0)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
