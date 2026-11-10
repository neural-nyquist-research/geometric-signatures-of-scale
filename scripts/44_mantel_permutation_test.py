#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Permutation test for the Mantel profile, and a partial Mantel control.

Why this script exists
----------------------
`01_Fig01_fig-mantel_analysis_comprehensive.py` computes r_M for every layer by
correlating the upper triangle of the neural distance matrix with that of the
typological matrix. That gives a point estimate and nothing else: no p-value, no
control for how many dimensions survive at a layer. An earlier draft of the paper
proposed a 9999-permutation test and a partial Mantel control, neither of which
the code performed. This script performs both, so that the paper can state them
and a reader can reproduce them.

What is computed, per model and layer
-------------------------------------
    r_M              standardised Mantel statistic, identical to Pearson's r over
                     the off-diagonal distance pairs (the equation in the paper).
    p_one_sided      (1 + #{r_perm >= r_obs}) / (1 + B), the direction of interest
                     being positive alignment.
    p_two_sided      two-sided permutation p, reported for completeness.
    r_partial        partial Mantel statistic controlling for the number of
                     retained dimensions per language at that layer. The control
                     matrix is |c_i - c_j| where c_i is the count of dimensions of
                     language i with |Max_Corr| > 0.1 at that layer, read from
                     `formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv`.
                     A layer-level scalar cannot serve as a Mantel control (it is
                     constant across pairs and the partial correlation is undefined),
                     so the per-language count is the only defensible version of
                     "controls for the number of retained dimensions at that layer".
    p_partial        permutation p for r_partial.

The permutation permutes the language labels of D_typ (rows and columns jointly),
which is the scheme the paper describes; the neural matrix is held fixed.

Reproducing the cache first
---------------------------
    --check compares r_M against ./re-organized/mantel_results_cache.csv for the
    canonical variant (Norm=True, cosine). If those disagree by more than 1e-9 the
    run stops: it means the activations or the typology table changed since the
    cache was written, and the permutation results would not belong to the numbers
    in the paper.

Usage
-----
    python 14_mantel_permutation_test.py --models xls-r-2b --layers 25,38
    python 14_mantel_permutation_test.py                       # all four, all layers
    python 14_mantel_permutation_test.py --n-perm 9999 --out ./re-organized/mantel_perm_results.csv

Run it from the directory that contains ./re-organized/.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

MODELS = {
    "xls-r-2b":   ("wav2vec2-xls-r-2b",      49),
    "xls-r-1b":   ("wav2vec2-xls-r-1b",      49),
    "xls-r-300m": ("wav2vec2-xls-r-300m",    25),
    "xlsr-53":    ("wav2vec2-large-xlsr-53", 25),
}

# The 24 FLEURS languages used by 01_Fig01; the dialect is not part of the
# typological matrix this script correlates against.
FOLDER_TO_ISO = {
    "arabic": "arb", "cantonese": "yue", "dutch": "nld", "english": "eng",
    "estonian": "ekk", "finnish": "fin", "french": "fra", "georgian": "kat",
    "german": "deu", "indonesian": "ind", "italian": "ita", "japanese": "jpn",
    "korean": "kor", "mandarin": "cmn", "polish": "pol", "portuguese": "por",
    "russian": "rus", "spanish": "spa", "swahili": "swh", "tamil": "tam",
    "thai": "tha", "turkish": "tur", "vietnamese": "vie", "zulu": "zul",
}

FEATURE_COLS = ["Syllable Structure", "Tone", "Consonant Inventories",
                "Vowel Quality Inventories", "voiced_ratio"]

HERE = os.path.dirname(os.path.abspath(__file__))
TYPOLOGY_CANDIDATES = [
    os.path.join(HERE, "..", "data", "target_languages_typology.csv"),
    os.path.join("data", "target_languages_typology.csv"),
    os.path.join("cldf-datasets_wals", "target_languages_typology.csv"),
    os.path.join("paper_revise", "target_languages_typology.csv"),
    "target_languages_typology.csv",
]

CACHE = os.path.join("re-organized", "mantel_results_cache.csv")
CANONICAL = "Norm=True, Metric=Cosine"


def find_typology_csv():
    for p in TYPOLOGY_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("typology CSV not found; looked in:\n  " + "\n  ".join(TYPOLOGY_CANDIDATES))


def upper(m, i, j):
    return m[i, j]


def partial_r(dab, dac, dbc):
    """Partial Mantel correlation of A and B given C, from three matrix correlations."""
    den = np.sqrt((1.0 - dac ** 2) * (1.0 - dbc ** 2))
    return np.nan if den == 0 else (dab - dac * dbc) / den


def retained_counts(model_dir, n_layers):
    """Per-language count of retained dimensions (|Max_Corr| > 0.1) at each layer."""
    path = os.path.join(model_dir, "formant_dim_analysis_cache",
                        "Formant_Lag_Analysis_Results_moreLayers.csv")
    out = {}
    if not os.path.exists(path):
        return out
    keep = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if abs(float(row["Max_Corr"])) <= 0.1:
                continue
            key = (int(row["Layer"]), row["Language"].strip().lower())
            keep.setdefault(key, set()).add(int(row["Dimension"]))
    for (layer, lang), dims in keep.items():
        out.setdefault(layer, {})[lang] = len(dims)
    return out


def compute(model_key, layers, n_perm, rng):
    folder, n_layers = MODELS[model_key]
    base = os.path.join("re-organized", folder)
    typ = pd.read_csv(find_typology_csv(), encoding="utf-8-sig")
    typ["ISO639P3code"] = typ["ISO639P3code"].astype(str).str.strip()
    counts = retained_counts(base, n_layers)

    rows = []
    for L in layers:
        protos, feats, langs = [], [], []
        for prefix, iso in FOLDER_TO_ISO.items():
            f = os.path.join(base, f"hidden_states_{L:02d}", f"{prefix}_fleurs",
                             f"hiddenlayer={L}", "X_flat_stable_1hours.npy")
            if not os.path.exists(f):
                continue
            sel = typ[typ["ISO639P3code"] == iso]
            if sel.empty:
                continue
            protos.append(np.load(f, mmap_mode="r").mean(axis=0))
            feats.append(sel[FEATURE_COLS].values[0])
            langs.append(prefix)
        if len(langs) < 10:
            print(f"  [skip] {model_key} L{L}: only {len(langs)} languages with both files")
            continue

        P = np.asarray(protos, dtype=np.float64)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12      # canonical: Norm=True
        D_neu = squareform(pdist(P, metric="cosine"))
        D_typ = squareform(pdist(np.asarray(feats, dtype=np.float64), metric="euclidean"))

        # Control matrix: per-language retained-dimension count at this layer.
        c_lang = counts.get(L, {})
        if len(c_lang) == len(langs) and all(l in c_lang for l in langs):
            c = np.array([c_lang[l] for l in langs], dtype=np.float64)
            D_ctl = np.abs(c[:, None] - c[None, :])
        else:
            D_ctl = np.zeros_like(D_neu)

        iu = np.triu_indices(len(langs), k=1)
        a = D_neu[iu]
        b0 = D_typ[iu]
        c0 = D_ctl[iu]

        r_obs = float(np.corrcoef(a, b0)[0, 1])
        r_ac = float(np.corrcoef(a, c0)[0, 1]) if c0.std() > 0 else np.nan
        r_bc = float(np.corrcoef(b0, c0)[0, 1]) if c0.std() > 0 else np.nan
        r_par = partial_r(r_obs, r_ac, r_bc) if np.isfinite(r_ac) else np.nan

        ge = gt = 0
        par_ge = 0
        n_par = 0
        n = len(langs)
        for _ in range(n_perm):
            p = rng.permutation(n)                     # permute D_typ rows/cols jointly
            b = D_typ[np.ix_(p, p)][iu]
            r = float(np.corrcoef(a, b)[0, 1])
            if r >= r_obs:
                ge += 1
            if abs(r) >= abs(r_obs):
                gt += 1
            if np.isfinite(r_par):
                rbc = float(np.corrcoef(b, c0)[0, 1]) if c0.std() > 0 else np.nan
                if np.isfinite(rbc):
                    rp = partial_r(r, r_ac, rbc)
                    n_par += 1
                    if rp >= r_par:
                        par_ge += 1

        rows.append(dict(Model=model_key, Layer=L, N_lang=n,
                         r_M=round(r_obs, 6),
                         p_one_sided=round((1 + ge) / (1 + n_perm), 5),
                         p_two_sided=round((1 + gt) / (1 + n_perm), 5),
                         r_partial=None if not np.isfinite(r_par) else round(r_par, 6),
                         p_partial=None if n_par == 0 else round((1 + par_ge) / (1 + n_par), 5)))
        print(f"  {model_key} L{L:<2d} n={n:<2d} r_M={r_obs:+.4f} p1={rows[-1]['p_one_sided']:.4f} "
              f"r_partial={rows[-1]['r_partial']}")
    return rows


def check_against_cache(rows):
    if not os.path.exists(CACHE):
        print(f"[check] no cache at {CACHE}; skipped")
        return True
    df = pd.read_csv(CACHE)
    df = df[df["Label"] == CANONICAL]
    bad = []
    for r in rows:
        m = df[(df["Model"] == r["Model"]) & (df["Layer"] == r["Layer"])]
        if m.empty:
            bad.append((r["Model"], r["Layer"], "absent from cache"))
            continue
        cached = float(m["Mantel_R"].iloc[0])
        if abs(cached - r["r_M"]) > 1e-6:
            bad.append((r["Model"], r["Layer"], f"cache {cached:.6f} vs now {r['r_M']:.6f}"))
    if bad:
        print("[check] FAILED -- the recomputation does not reproduce the cache:")
        for b in bad[:10]:
            print("   ", b)
        return False
    print(f"[check] OK -- {len(rows)} layer(s) reproduce the canonical column of the cache")
    return True


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS), help="default: all four")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices, e.g. 13,19,25,38 "
                         "(default: every layer of each model)")
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", default=os.path.join("re-organized", "mantel_perm_results.csv"))
    ap.add_argument("--check", action="store_true",
                    help="verify r_M against mantel_results_cache.csv and stop on mismatch")
    ap.add_argument("--no-check", action="store_true", help="skip that verification")
    a = ap.parse_args()

    # The hidden states are not shipped with the companion repository; stop with a
    # readable message rather than a FileNotFoundError deep in the layer loop.
    probe = os.path.join("re-organized", MODELS[a.models[0]][0], "hidden_states_13",
                         "english_fleurs", "hiddenlayer=13", "X_flat_stable_1hours.npy")
    if os.path.isdir(os.path.join("re-organized", MODELS[a.models[0]][0])) and not os.path.exists(probe):
        print("[ERROR] hidden states are not present in this checkout.")
        print("        This script runs where the activations were extracted; the published output")
        print("        is data/mantel_perm_results.csv, shipped with the repository.")
        return 3

    rng = np.random.default_rng(a.seed)
    all_rows = []
    for key in a.models:
        _, n_layers = MODELS[key]
        layers = ([int(x) for x in a.layers.split(",")] if a.layers
                  else list(range(n_layers)))
        print(f">>> {key}: {len(layers)} layer(s), {a.n_perm} permutations")
        all_rows += compute(key, layers, a.n_perm, rng)

    if not all_rows:
        print("[ERROR] nothing computed; check that ./re-organized/ is present")
        return 2

    if a.check and not check_against_cache(all_rows):
        return 3

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {a.out}  ({len(all_rows)} rows)")
    print("Next: quote p_one_sided and r_partial for the layers the paper reads "
          "(L13/L25 anchor, L18/L19/L38 late, plus each model's global maximum).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
