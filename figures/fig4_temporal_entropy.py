#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-plot the temporal-entropy figure at final print size (7.16 in, 2 stacked panels).

The original PNG was 3259x3866 px (10.9x12.9 in at 300 dpi) and was included at
\\linewidth inside a single-column figure (~3.5 in), i.e. a 0.32x reduction that left
~4 pt type.  This script redraws both panels from the source CSVs at 1:1 print size.

Panel A: normalised temporal entropy H_norm(L) = H_t(L) / log2(D) over relative depth.
Panel B: distribution of best lags at the mid-depth anchor (L13/L13/L25/L25).

Data:  re-organized/<model>/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv
       significant dimensions only (|Max_Corr| > 0.1), lag bins of 50 ms
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
OUT = os.path.join(HERE, "fig4_temporal_entropy.png")

MODELS = [
    ("XLSR-53",    "wav2vec2-large-xlsr-53", 1024, "#E69F00"),
    ("XLS-R-300M", "wav2vec2-xls-r-300m",    1024, "#56B4E9"),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b",      1280, "#009E73"),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b",      1920, "#0072B2"),
]
DT = 0.05           # bin width, seconds (see Methods)
ANCHOR = {"XLSR-53": 13, "XLS-R-300M": 13, "XLS-R-1B": 25, "XLS-R-2B": 25}


def load_lags(model_dir):
    path = os.path.join(ROOT, model_dir, "formant_dim_analysis_cache",
                        "Formant_Lag_Analysis_Results_moreLayers.csv")
    per_layer = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mc = float(row["Max_Corr"])
            if abs(mc) > 0.1:
                per_layer.setdefault(int(row["Layer"]), []).append(float(row["Best_Lag_Sec"]))
    return per_layer


def entropy(lags, dt=DT):
    if len(lags) < 2:
        return np.nan
    lo, hi = lags.min(), lags.max()
    nbins = max(1, int(np.ceil((hi - lo) / dt)))
    counts, _ = np.histogram(lags, bins=nbins, range=(lo, lo + nbins * dt))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.55), dpi=300,
                                   gridspec_kw=dict(height_ratios=[1.05, 1.0]))

    for name, mdir, dim, colour in MODELS:
        per_layer = load_lags(mdir)
        layers = sorted(per_layer)
        tmax = max(layers)
        rel = np.array([L / tmax for L in layers])
        h = np.array([entropy(np.asarray(per_layer[L])) for L in layers])
        hnorm = h / np.log2(dim)

        ax1.plot(rel, hnorm, "-", color=colour, linewidth=1.15, label=name)
        if rel.size > 120:                       # keep markers sparse on 49-layer models
            step = 4
            ax1.plot(rel[::step], hnorm[::step], "o", color=colour, markersize=2.2)

        # mark the mid-depth anchor only; no peak annotation (see caption note)
        L_bt = ANCHOR[name]
        k_bt = layers.index(L_bt)
        ax1.plot(rel[k_bt], hnorm[k_bt], "*", color=colour, markersize=8,
                 markeredgecolor="white", markeredgewidth=0.4, zorder=6)

        lags = np.asarray(per_layer[layers[k_bt]])
        ax2.hist(lags, bins=np.arange(-0.85, 0.8501, 0.10), density=True, histtype="step",
                 linewidth=1.15, color=colour,
                 label=f"{name} (L{layers[k_bt]}), H$_t$={entropy(lags):.2f} bits")

    ax1.set_xlabel("Relative depth  (0 = input, 1 = output)", fontsize=7.5)
    ax1.set_ylabel(r"$\mathcal{H}_{\mathrm{norm}} = H_t / \log_2 D$", fontsize=7.5)
    ax1.set_title("Normalised temporal entropy across depth", fontsize=8.2, pad=4)
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(alpha=0.22, linewidth=0.5)
    ax1.tick_params(labelsize=6.3, length=2, pad=1.5)
    ax1.legend(fontsize=6.2, loc="upper center", ncol=4, framealpha=0.92,
               borderpad=0.3, handlelength=1.4, columnspacing=1.2)
    ax1.text(0.985, 0.03, "stars: mid-depth anchor (L13 / L13 / L25 / L25)",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=5.6, color="#555555")
    ax1.set_ylim(0.02, 0.42)

    ax2.set_xlabel("Best lag  $\\tau$ (s)", fontsize=7.5)
    ax2.set_ylabel("Probability density", fontsize=7.5)
    ax2.set_title("Functional lag distribution at each model's mid-depth anchor (100 ms bins)", fontsize=8.2, pad=4)
    ax2.set_xlim(-0.85, 0.85)
    ax2.grid(alpha=0.22, linewidth=0.5)
    ax2.tick_params(labelsize=6.3, length=2, pad=1.5)
    ax2.legend(fontsize=5.8, loc="upper left", framealpha=0.9, borderpad=0.3,
               handlelength=1.6, labelspacing=0.25)

    fig.tight_layout(h_pad=1.1)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
