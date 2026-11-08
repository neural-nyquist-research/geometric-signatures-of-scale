#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-layer Mantel correlation between neural and typological distances,
computed for all four XLS-R models and drawn one figure per model.

What it does
------------
For every layer of a model, take the mean activation of each language
(`X_flat_stable_1hours.npy`, mmapped), build the neural distance matrix between
language prototypes and the typological distance matrix from the WALS/PHOIBLE
feature table, and correlate the two (Mantel r). This is done for four
extraction variants (L2-normalised or not x cosine or euclidean).


Cache policy -- the family is drawn only when it is complete
------------------------------------------------------------
One cache holds every model: ./re-organized_mls/mantel_results_cache.csv with
columns `Model, Layer, Mantel_R, Label`. It is a single file, not one per model,
because the figure is a family comparison: a single file makes "are all four
present?" a one-line query and keeps the four extraction variants consistent.

Per-model files from the earlier four-file layout
(./re-organized_mls/mantel_results_cache_<nickname>.csv) are imported automatically,
so results already computed that way are not thrown away. The import is one-shot
per model: as soon as a model is `complete` in the unified cache its legacy file
is skipped and the unified cache is not rewritten, so an interrupted run that is
restarted does no redundant I/O. Unified rows win over legacy rows on a
(Model, Label, Layer) clash, and legacy rows whose Label is not one of the four
expected variants are dropped rather than injected as extra curves.

Each model is classified as

    complete : all four extraction variants present over its full layer range
    partial  : cache exists but an interrupted run left rows missing
    absent   : nothing cached

and main() runs in two phases:

    phase 1  one pass over the requested models. `complete` is reused; `partial`
             and `absent` are computed. A model whose raw activations or the
             typology CSV cannot be reached is skipped ([skip]) without
             aborting the remaining models.
    phase 2  print a status table, then draw the figures only if EVERY requested
             model is `complete`. Otherwise nothing is drawn and the exit code is
             1, so the run stays resumable: finish the missing models and re-run.

Rationale: a partial cache would silently produce a family figure with curves
missing, which is exactly the failure mode this figure is meant to avoid.

--plot-partial and --plot-only are *render* requests: they never start a scan, so
with a cache that already holds some models you get those figures immediately.
--plot-partial draws whatever is complete; --plot-only does the same but still
refuses to draw unless every requested model is complete. Add --recompute if you
want a compute run that ignores the cache.

Completeness is judged on row counts and variant coverage only: the cache is
keyed by model name and layer range, so it cannot notice that the underlying
activations or the typology table have changed. Re-run with --recompute after
re-extracting either.

Exit codes: 0 = figures drawn, 1 = something is still missing (nothing drawn).


Differences from 01_Fig01_fig-mantel_analysis.py (single-model original)
------------------------------------------------------------------------
1. One cache for the whole family with a `Model` column, so a run resumes per
   model instead of redoing every scan. A legacy single-model cache (no `Model`
   column) is accepted and attributed to xls-r-2b.
2. Plotting selects rows by model name (`df[df.Model == nickname]`), so each
   model gets its own figure instead of overwriting one shared file.
3. Annotation landmarks are derived from each model's own curve (find_landmarks:
   early local maximum, plateau entry, global maximum, terminal collapse) rather
   than hard-coded at L6/L16/L22/L38, which only make sense for the 48-layer 2B
   model. Plateau entry uses a moving average with a raw-value fallback, so a
   single-spike curve degrades gracefully instead of reporting L0.

Fixes carried along
-------------------
* `savefig(f"./re-organized_mls/{{nickname}}_typology_final.png")` wrote the literal
  string "{nickname}", so all four models overwrote one file and only the last
  survived; now a real f-string.
* Layer index range is per model (25 for the 300M pair, 49 for 1B/2B) instead of
  a fixed range(49), which made the smaller models probe layers that do not exist.
* Typology CSV is read with utf-8-sig (the distributed file carries a BOM, which
  otherwise lands in the first column name).
* find_typology_csv() used to be called outside the per-model try block, so a
  missing typology CSV aborted the whole run instead of skipping one model.
* --plot-only used to discard the loaded cache, leaving nothing to plot.


Usage
-----
    # first run: computes whatever is missing, then draws if all four are done
    python 01_Fig01_fig-mantel_analysis_comprehensive.py

    # resume after an interruption -- complete models are reused
    python 01_Fig01_fig-mantel_analysis_comprehensive.py

    # only some models
    python 01_Fig01_fig-mantel_analysis_comprehensive.py --models xls-r-2b xlsr-53

    # force a recomputation (cache still merged, never duplicated)
    python 01_Fig01_fig-mantel_analysis_comprehensive.py --recompute

    # re-render from the existing cache without touching raw activations
    python 01_Fig01_fig-mantel_analysis_comprehensive.py --plot-only

    # render only what is already cached -- never starts a scan, so it returns at once
    python 01_Fig01_fig-mantel_analysis_comprehensive.py --plot-partial

    # figure sized for an IEEEtran column (7.16 in) instead of 14 in screen size
    python 01_Fig01_fig-mantel_analysis_comprehensive.py --print-size

    # --show opens each figure; --no-log skips the .log file

The script must run from the directory that contains ./re-organized_mls/ (the cache
and the figures are written there), e.g. the notebooks folder on the analysis
machine.


Outputs
-------
    ./re-organized_mls/mantel_results_cache.csv          Model, Layer, Mantel_R, Label
                                                     (single unified cache; legacy
                                                     per-model files are merged in
                                                     and left untouched on disk)
    ./re-organized_mls/<nickname>_typology_final.png     one figure per model
    ./01_Fig01_fig-mantel_analysis_comprehensive.log tee of stdout
"""
import argparse
import gc
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# Logging: mirror stdout to a .log next to this script
# ══════════════════════════════════════════════════════════════════════
class _Tee:
    """Write stdout to both screen and file."""

    def __init__(self, file_path):
        self._terminal = sys.stdout
        self._log = open(file_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def isatty(self):
        return False

    def close(self):
        self._log.close()
        sys.stdout = self._terminal


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
# nickname -> (folder under ./re-organized_mls/, number of layer indices)
MODELS = {
    "xls-r-2b":   ("wav2vec2-xls-r-2b",      49),
    "xls-r-1b":   ("wav2vec2-xls-r-1b",      49),
    "xls-r-300m": ("wav2vec2-xls-r-300m",    25),
    "xlsr-53":    ("wav2vec2-large-xlsr-53", 25),
}

CACHE_FILE = "./re-organized_mls/mantel_results_cache.csv"
CACHE_COLUMNS = ["Model", "Layer", "Mantel_R", "Label"]

# a cache entry only counts as usable when the model has all four extraction
# variants over its full layer range -- an interrupted run leaves a partial file
EXPECTED_LABELS = (
    "Norm=True, Metric=Cosine",
    "Norm=True, Metric=Euclidean",
    "Norm=False, Metric=Cosine",
    "Norm=False, Metric=Euclidean",
)

TYPOLOGY_CANDIDATES = [
    "./cldf-datasets_wals/target_languages_typology.csv",
    "./paper_revise/target_languages_typology.csv",
    "./data/target_languages_typology.csv",
    "./target_languages_typology.csv",
]

TARGET_LABEL = "Norm=True, Metric=Cosine"   # curve used for annotation

GRAY_MID = "#8B949E"
GRAY_LIGHT = "#D1D5DA"
COLLAPSE_RED = "#FF7B72"
PLATEAU_BLUE = "#2188ff"
PLATEAU_DARK = "#0366d6"

FIG_SIZE = (14, 8)          # screen default, as in the original
FIG_SIZE_PRINT = (7.16, 4.2)  # column width of an IEEEtran two-column page
DPI = 300

folder_to_iso = {
    'dutch': 'nld', 'english': 'eng',
    'french': 'fra', 'german': 'deu',
    'italian': 'ita', 'polish': 'pol',
    'portuguese': 'por', 'spanish': 'spa',
}

feature_cols = ['Syllable Structure', 'Tone', 'Consonant Inventories',
                'Vowel Quality Inventories', 'voiced_ratio']


# ══════════════════════════════════════════════════════════════════════
# Cache helpers
# ══════════════════════════════════════════════════════════════════════
def read_cache(path=CACHE_FILE):
    """Return the cache as a DataFrame, tolerating the legacy single-model format."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Model" not in df.columns:
        # legacy cache: one model only, presumably the 2B model it was written for
        df["Model"] = "xls-r-2b"
        print(f"[cache] legacy format detected, attributing existing rows to 'xls-r-2b'")
    missing = [c for c in CACHE_COLUMNS if c not in df.columns]
    if missing:
        print(f"[cache] unexpected columns (missing {missing}); recomputing from scratch")
        return None
    return df[CACHE_COLUMNS]


LEGACY_GLOB = "./re-organized_mls/mantel_results_cache_*.csv"


def import_legacy_caches(path=CACHE_FILE, unified=None):
    """Fold per-model cache files from the earlier four-file layout into one frame.

    The first comprehensive version wrote ./re-organized_mls/mantel_results_cache_<nickname>.csv
    (one file per model). Those files hold real computed results, so they are picked
    up instead of being thrown away by the unified cache.

    The import is one-shot per model: a model that is already `complete` in the
    unified cache is skipped, so re-running after an interruption neither re-reads
    the file nor rewrites the unified cache. Rows whose Label is not one of the four
    expected variants are dropped, so a half-written legacy file cannot inject extra
    curves into the figure.

    Returns a DataFrame with CACHE_COLUMNS, or None when there is nothing to import.
    """
    files = sorted(glob.glob(LEGACY_GLOB))
    if not files:
        return None

    frames = []
    for f in files:
        name = os.path.basename(f)
        nick = name[len("mantel_results_cache_"):-len(".csv")]
        if nick not in MODELS:
            print(f"[legacy] {name}: unknown model '{nick}', ignored")
            continue
        if unified is not None and cache_status(unified, nick) == "complete":
            print(f"[legacy] {name}: {nick} already complete in the unified cache, "
                  f"not re-imported")
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[legacy] {name}: unreadable ({e}), ignored")
            continue
        missing = [c for c in ("Layer", "Mantel_R", "Label") if c not in df.columns]
        if missing:
            print(f"[legacy] {name}: missing {missing}, ignored")
            continue

        df = df.copy()
        df["Model"] = nick
        df = df[CACHE_COLUMNS]
        extra = ~df["Label"].isin(EXPECTED_LABELS)
        if extra.any():
            print(f"[legacy] {name}: dropped {int(extra.sum())} row(s) with an "
                  f"unexpected Label ({', '.join(sorted(set(df.loc[extra, 'Label'])))[:60]})")
            df = df[~extra]
        if df.empty:
            print(f"[legacy] {name}: nothing usable, ignored")
            continue

        frames.append(df)
        print(f"[legacy] imported {len(df):4d} rows for {nick:11s} from {name}")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_cache(path=CACHE_FILE):
    """Unified cache, with any per-model legacy files merged in (unified rows win)."""
    unified = read_cache(path)
    legacy = import_legacy_caches(path, unified)

    if legacy is None:
        return unified
    if unified is None or unified.empty:
        merged = legacy
    else:
        merged = pd.concat([unified, legacy], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["Model", "Label", "Layer"], keep="first")
    dropped = before - len(merged)
    print(f"[cache] legacy merge: {before} rows -> {len(merged)} "
          f"({dropped} duplicates dropped)")
    # persist the merged view so the next run needs no legacy scan
    merged = merged.sort_values(["Model", "Label", "Layer"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def cache_status(cache, nickname):
    """Classify a model's cache as 'complete', 'partial' or 'absent'.

    'complete' means every expected extraction variant is present and covers the
    model's full layer range, so the figure can be drawn without recomputation.
    """
    if cache is None or cache.empty:
        return "absent"
    sub = cache[cache["Model"] == nickname]
    if sub.empty:
        return "absent"
    n_layers = MODELS[nickname][1]
    per_label = sub.groupby("Label")["Layer"].nunique()
    have = [int(per_label.get(lab, 0)) for lab in EXPECTED_LABELS]
    return "complete" if min(have) >= n_layers else "partial"


def write_cache(new_rows, path=CACHE_FILE):
    """Merge `new_rows` into the cache, replacing any rows of the same models."""
    old = read_cache(path)
    if old is not None and not old.empty:
        keep = old[~old["Model"].isin(new_rows["Model"].unique())]
        merged = pd.concat([keep, new_rows], ignore_index=True)
    else:
        merged = new_rows.copy()
    merged = merged.sort_values(["Model", "Label", "Layer"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    print(f"[cache] {len(merged)} rows written to {path} "
          f"(models: {', '.join(sorted(merged['Model'].unique()))})")
    return merged


# ══════════════════════════════════════════════════════════════════════
# Computation
# ══════════════════════════════════════════════════════════════════════
def find_typology_csv():
    """First existing typology CSV among TYPOLOGY_CANDIDATES.

    Raises FileNotFoundError (listing every path tried) so the caller can skip a
    single model instead of aborting the run.
    """
    for p in TYPOLOGY_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "typology CSV not found; looked in:\n  " + "\n  ".join(TYPOLOGY_CANDIDATES))


def precompute_layer_prototypes(base_dir, layers, df_typeo):
    """Language-mean prototypes for every layer (one pass over the mmapped arrays)."""
    layer_cache = {}
    print(f">>> Reading raw activations from {base_dir} ({len(layers)} layers)")
    for l_str in tqdm(layers, desc="Layers"):
        layer_dir = os.path.join(base_dir, f"hidden_states_{l_str}")
        protos_raw, feats_raw = [], []

        for folder_prefix, iso in tqdm(folder_to_iso.items(), desc="Languages", leave=False):
            path_part = f"hiddenlayer={int(l_str)}"
            file_path = os.path.join(layer_dir, f"{folder_prefix}_mls",
                                     path_part, "X_flat_stable_1hours.npy")
            if not os.path.exists(file_path):
                continue
            data = np.load(file_path, mmap_mode="r")     # keeps memory pressure low
            proto = np.mean(data, axis=0)

            typeo_data = df_typeo[df_typeo["ISO639P3code"] == iso][feature_cols].values
            if len(typeo_data) > 0:
                protos_raw.append(proto)
                feats_raw.append(typeo_data[0])
            del data

        layer_cache[l_str] = {"protos": np.array(protos_raw), "feats": np.array(feats_raw)}
        gc.collect()
    return layer_cache


def compute_model(nickname, df_typeo=None):
    """Compute Mantel r for one model; returns a DataFrame with a Model column."""
    folder, n_layers = MODELS[nickname]
    base_dir = f"./re-organized_mls/{folder}/"
    layers = [f"{i:02d}" for i in range(n_layers)]

    if df_typeo is None:
        df_typeo = pd.read_csv(find_typology_csv(), encoding="utf-8-sig")
        df_typeo["ISO639P3code"] = df_typeo["ISO639P3code"].astype(str).str.strip()

    print(f">>> Computing {nickname} ({n_layers} layers) from {base_dir}")
    print(f"    each layer reads {len(folder_to_iso)} language files; the tqdm bar "
          f"gives the ETA after the first layer")
    t_start = time.time()
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"{base_dir} does not exist")

    layer_data_cache = precompute_layer_prototypes(base_dir, layers, df_typeo)

    rows = []
    for use_l2 in tqdm([True, False], desc="L2Norm"):
        for metric in tqdm(["cosine", "euclidean"], desc="Distance", leave=False):
            label = f"Norm={use_l2}, Metric={metric.capitalize()}"
            for l_str in tqdm(layers, desc="Layers", leave=False):
                d = layer_data_cache[l_str]
                if len(d["protos"]) < 2:  # < 10
                    continue
                p_work = d["protos"].copy()
                if use_l2:
                    p_work = normalize(p_work, axis=1, norm="l2")

                d_speech = squareform(pdist(p_work, metric=metric))
                d_typeo = squareform(pdist(d["feats"], metric="euclidean"))
                idx = np.triu_indices(len(d_speech), k=1)
                r, _ = pearsonr(d_speech[idx], d_typeo[idx])
                rows.append({"Model": nickname, "Layer": int(l_str),
                             "Mantel_R": float(r), "Label": label})

    df = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    mins = (time.time() - t_start) / 60.0
    print(f">>> {nickname}: {len(df)} rows in {mins:.1f} min")
    return df


# ══════════════════════════════════════════════════════════════════════
# Landmarks derived from the curve itself
# ══════════════════════════════════════════════════════════════════════
def find_landmarks(layers, values):
    """Locate the features worth annotating, from the curve alone.

    Returns a dict with
        early      : layer of the first local maximum before the peak
        plateau    : first layer of the sustained high region
        peak       : layer of the global maximum
        collapse   : first layer of the terminal low run (None if there is none)
        r_max, r_min
    """
    layers = np.asarray(layers)
    v = np.asarray(values, dtype=float)
    n = len(v)
    out = {"early": None, "plateau": None, "peak": None, "collapse": None,
           "r_max": float(np.nanmax(v)), "r_min": float(np.nanmin(v))}
    if n < 3:
        return out

    r_max, r_min = out["r_max"], out["r_min"]
    rng = r_max - r_min if r_max > r_min else 1.0

    peak_i = int(np.nanargmax(v))
    out["peak"] = int(layers[peak_i])

    # plateau: earliest layer from which the local average stays near the maximum
    level = r_max - 0.15 * rng
    w = max(3, n // 8)
    smooth = np.convolve(v, np.ones(w) / w, mode="same")
    plateau_i = None
    for i in range(peak_i + 1):
        if smooth[i] >= level:
            plateau_i = i
            break
    if plateau_i is None:
        # a single sharp spike is diluted by the moving average; fall back to the
        # first raw layer that clears the level (v[peak_i] always does)
        above = np.where(v[: peak_i + 1] >= level)[0]
        plateau_i = int(above[0]) if len(above) else max(0, peak_i - 1)
    out["plateau"] = int(layers[plateau_i])

    # early: first local maximum before the peak (fallback: first strong rise)
    early_i = None
    for i in range(1, max(2, peak_i)):
        if v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            early_i = i
            break
    if early_i is None:
        above = np.where(v[: max(1, peak_i)] >= r_min + 0.5 * rng)[0]
        early_i = int(above[0]) if len(above) else 0
    out["early"] = int(layers[early_i])

    # collapse: terminal run below max - 60% of the range
    low = r_max - 0.60 * rng
    j = n - 1
    while j >= 0 and v[j] < low:
        j -= 1
    if j < n - 1:
        out["collapse"] = int(layers[j + 1])
    return out


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════
def plot_model(res_df, nickname, fig_size=FIG_SIZE, show=False, out_dir="./re-organized_mls"):
    """Draw the four extraction variants for one model and annotate it dynamically."""
    df = res_df[res_df["Model"] == nickname].sort_values("Layer")
    if df.empty:
        print(f"[plot] no rows for {nickname}, skipped")
        return None

    n_layers = int(df["Layer"].max()) + 1
    best = df[df["Label"] == TARGET_LABEL]
    if best.empty:
        best = df[df["Label"] == df["Label"].unique()[0]]
        print(f"[plot] {nickname}: '{TARGET_LABEL}' absent, annotating '{best['Label'].iloc[0]}'")

    layers = best["Layer"].to_numpy()
    values = best["Mantel_R"].to_numpy()
    lm = find_landmarks(layers, values)
    r_max, r_min = lm["r_max"], lm["r_min"]
    rng = (r_max - r_min) or 1.0

    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(data=df, x="Layer", y="Mantel_R", hue="Label", style="Label",
                 markers=True, dashes=[(None, None), (2, 2), (5, 5), (1, 1)],
                 linewidth=2.0, alpha=0.85, ax=ax)

    # --- plateau band, from the curve's own sustained region -------------
    if lm["collapse"] is not None:
        band_end = lm["collapse"]
    else:
        band_end = int(layers[-1])
    if lm["plateau"] is not None and band_end > lm["plateau"]:
        ax.axvspan(lm["plateau"], band_end, color=PLATEAU_BLUE, alpha=0.06, zorder=0)
        # bottom of the axes, inside the band: never collides with the peak labels
        ax.text((lm["plateau"] + band_end) / 2, r_min - 0.09 * rng,
                "sustained\ntypological retention", ha="center", va="bottom",
                fontsize=7.5 if fig_size[0] < 9 else 10, color=PLATEAU_DARK,
                fontweight="bold", linespacing=1.1)

    # --- terminal collapse ----------------------------------------------
    if lm["collapse"] is not None:
        ax.axvspan(lm["collapse"], layers[-1] + 0.5, color=COLLAPSE_RED, alpha=0.15,
                   zorder=0, label="Late collapse")
        # the band is only a few layers wide, so the label is rotated to fit it
        ax.text((lm["collapse"] + layers[-1]) / 2 + 0.15,
                (r_min + r_max) / 2, "late collapse", color="#D73A49",
                ha="center", va="center", rotation=90,
                fontsize=7.5 if fig_size[0] < 9 else 9.5, fontweight="bold")

    # --- point annotations ------------------------------------------------
    first_layer, last_layer = int(layers[0]), int(layers[-1])
    span = max(1, last_layer - first_layer)

    def annotate(layer, text, above=True):
        row = best[best["Layer"] == layer]
        if row.empty:
            return
        y = float(row["Mantel_R"].iloc[0])
        off = 0.09 * rng if above else -0.11 * rng
        # keep the box inside the axes when the point sits near an edge
        frac = (layer - first_layer) / span
        ha = "left" if frac < 0.15 else ("right" if frac > 0.85 else "center")
        ax.annotate(text, xy=(layer, y), xytext=(layer, y + off),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY_MID, lw=1.2),
                    fontsize=9 if fig_size[0] < 9 else 11, fontweight="bold",
                    ha=ha, va="bottom" if above else "top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRAY_MID,
                              alpha=0.9, lw=0.8))

    if lm["early"] is not None:
        annotate(lm["early"], "Early acoustic encoding", above=False)
    if lm["plateau"] is not None and lm["plateau"] != lm["early"]:
        annotate(lm["plateau"], "Plateau entry", above=True)
    if lm["peak"] is not None:
        annotate(lm["peak"], f"Global maximum (r={r_max:.3f})", above=True)

    # --- axes -------------------------------------------------------------
    ax.set_ylim(r_min - 0.12 * rng, r_max + 0.22 * rng)
    step = 4 if n_layers > 40 else 2
    ax.set_xticks(range(0, n_layers, step))
    ax.tick_params(axis="x", labelsize=8 if fig_size[0] < 9 else 9)

    title_fs = 9 if fig_size[0] < 9 else 15
    ax.set_title(f"Cross-layer Mantel correlation — {nickname.upper()} "
                 f"({n_layers} layers)", fontsize=title_fs, pad=10, fontweight="bold")
    ax.set_xlabel("Transformer layer index", fontsize=8 if fig_size[0] < 9 else 12)
    ax.set_ylabel("Typological correlation (Mantel $r$)",
                  fontsize=8 if fig_size[0] < 9 else 12)
    ax.legend(title="Extraction parameters", bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=True, fontsize=7 if fig_size[0] < 9 else 10)

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"{nickname}_typology_final.png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    print(f"[plot] wrote {out_png}  "
          f"(peak L{lm['peak']} r={r_max:.3f}; plateau from L{lm['plateau']}; "
          f"collapse from L{lm['collapse']})")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# ══════════════════════════════════════════════════════════════════════
def main():
    """Two-phase driver: complete the cache, then draw only if the family is whole.

    Returns 0 when the figures were written, 1 when at least one requested model
    is still missing (nothing is drawn in that case, so the run can be resumed).
    """
    ap = argparse.ArgumentParser(
        description="Mantel correlation between neural and typological distances, "
                    "four XLS-R models.",
        epilog="Figures are drawn only when every requested model is complete; "
               "otherwise the run prints what is missing and exits 1 so it can be "
               "resumed. The module docstring documents the full cache policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS), help="subset of models to process")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore the cache and recompute from raw activations")
    ap.add_argument("--print-size", action="store_true",
                    help="render at column width (7.16 in) instead of 14 in")
    ap.add_argument("--show", action="store_true", help="also open each figure")
    ap.add_argument("--no-log", action="store_true", help="do not tee stdout to a .log")
    ap.add_argument("--plot-partial", action="store_true",
                    help="render only the models already complete in the cache and do "
                         "NOT compute anything; exits 1 if none are ready")
    ap.add_argument("--plot-only", action="store_true",
                    help="same as --plot-partial, but still refuses to draw unless every "
                         "requested model is complete")
    a = ap.parse_args()

    script_name = os.path.basename(__file__)
    log_filename = script_name[:-3] + ".log" if script_name.endswith(".py") else script_name + ".log"
    if not a.no_log:
        sys.stdout = _Tee(os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename))
        print(f"Log file will be saved as: {log_filename}")

    print(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {a.models}   recompute={a.recompute}")

    cache = load_cache()          # --recompute only forces the per-model status to "absent"
    df_typeo = None

    rc = 0
    try:
        # ---- phase 1: one pass over the requested models, compute only what is missing
        # --plot-partial and --plot-only are render requests: they never start a scan.
        skip_compute = (a.plot_only or a.plot_partial) and not a.recompute
        for nickname in a.models:
            st = cache_status(cache, nickname)
            if st == "complete" and not a.recompute:
                n_rows = int((cache["Model"] == nickname).sum())
                print(f"[cache] {nickname}: complete ({n_rows} rows), reused")
                continue
            if skip_compute:
                print(f"[cache] {nickname}: {st} -- not computed (render-only run)")
                continue

            if st == "partial":
                print(f"[cache] {nickname}: PARTIAL cache (interrupted run?), recomputing")
            else:
                print(f"[cache] {nickname}: absent, computing")

            try:
                if df_typeo is None:
                    df_typeo = pd.read_csv(find_typology_csv(), encoding="utf-8-sig")
                    df_typeo["ISO639P3code"] = df_typeo["ISO639P3code"].astype(str).str.strip()
                new = compute_model(nickname, df_typeo)
                cache = write_cache(new)
            except FileNotFoundError as e:
                print(f"[skip] {nickname}: {e}")
            except Exception as e:                   # keep going: other models may work
                import traceback
                traceback.print_exc()
                print(f"[fail] {nickname}: {e}")

        # ---- phase 2: decide whether the figure set may be drawn
        status = {m: cache_status(cache, m) for m in a.models}
        print("\n" + "=" * 62)
        print("cache status")
        for m in a.models:
            n = 0 if cache is None else int((cache["Model"] == m).sum())
            print(f"  {m:11s} {status[m]:9s} {n:6d} rows")
        print("=" * 62)

        ready = [m for m in a.models if status[m] == "complete"]
        pending = [m for m in a.models if status[m] != "complete"]

        if not ready:
            print("Nothing to plot: no model is complete yet.")
            print("Re-run (without flags) once the raw activations are reachable.")
            rc = 1
        elif pending and not a.plot_partial:
            # all-or-nothing by default: a partial cache would yield a family figure
            # with curves silently missing, which is the failure this figure avoids
            print(f"{len(pending)} of {len(a.models)} requested models are not complete: "
                  f"{', '.join(pending)}")
            print("Not drawing anything yet -- finish them and re-run, so the family is "
                  "always shown as a complete set.")
            print("Use --plot-partial to render the finished models anyway.")
            rc = 1
        else:
            if pending:
                print(f"[plot] --plot-partial: drawing {len(ready)} model(s); "
                      f"still missing {', '.join(pending)}")
            size = FIG_SIZE_PRINT if a.print_size else FIG_SIZE
            for nickname in ready:
                plot_model(cache, nickname, fig_size=size, show=a.show)

    except KeyboardInterrupt:
        print("\n[interrupt] stopped by user (Ctrl-C).")
        try:
            saved = {m: cache_status(cache, m) for m in a.models}
        except Exception:
            saved = {}
        done = [m for m, st in saved.items() if st == "complete"]
        print(f"[interrupt] complete in cache: {', '.join(done) if done else 'none'}")
        print("[interrupt] those models are saved and will NOT be recomputed; re-run the "
              "same command to resume.")
        print("[interrupt] a model interrupted mid-scan restarts from its first layer, "
              "because rows are written once the whole model is done.")
        rc = 130

    print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not a.no_log and hasattr(sys.stdout, "close"):
        sys.stdout.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
