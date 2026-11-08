#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pairwise capacity contrasts inside the XLS-R family, twelve relative depths.

The paper's headline contrast is 1B vs 2B, the only pair that isolates width at
fixed data (128 languages) and fixed depth (48 layers). That choice made the
paper's negative result sharp but narrow: it asks whether *doubling* from 1B to 2B
buys a consistent geometric gain, and answers no.

This script asks the prior question instead: does the 300M -> 1B and 300M -> 2B
step show a reproducible effect? XLSR-53 is excluded as a reference because it is
trained on 53 languages rather than 128, so it is a data contrast, not a capacity
one. All four pairs are still reported so the numbers can be read side by side.

Everything is computed with the same primitives as
02_07_formant_dim_statistical_test_comprehensive.py (imported, not re-implemented):
per-node |Max_Corr| and Best_Lag over dimensions with |R_max| > 0.1, per-language
intrinsic dimension, and the bootstrap difference of hull areas.

Output: one block per pair, then a summary table.

Usage
-----
    python 43_10_pairwise_capacity_pairs.py
    python 43_10_pairwise_capacity_pairs.py --loops 12 --out pair_summary.csv
"""

import argparse
import csv
import importlib.util
import os

def _project_root():
    """The directory holding re-organized/. Current directory, then the script's, then its parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = _project_root()


def load_engine():
    _cand = [os.path.join(PROJECT, "02_07_formant_dim_statistical_test_comprehensive.py"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "02_07_formant_dim_statistical_test_comprehensive.py")]
    path = next((p for p in _cand if os.path.exists(p)), _cand[0])
    spec = importlib.util.spec_from_file_location("engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PAIRS = [("XLS-R-300M", "XLS-R-1B", "capacity + depth"),
         ("XLS-R-300M", "XLS-R-2B", "capacity + depth"),
         ("XLS-R-1B", "XLS-R-2B", "width only (paper's contrast)"),
         ("XLSR-53", "XLS-R-300M", "training data (control)")]

METRICS = ["Max_Corr", "Best_Lag_Sec", "Intrinsic_Dim", "Area(diff)"]


def run(loops=12):
    E = load_engine()
    totals = E.model_total_layers()
    area_tbl = E.load_area_table()
    lang_floor = sorted(E.load_id(E.NAMES[0], 0).keys())

    res = {(a, b, m): [] for a, b, _k in PAIRS for m in METRICS}

    for k in range(1, loops + 1):
        layers = {n: E.round_layer(n, k, loops, totals) for n in E.NAMES}
        mc, bl, ids = {}, {}, {}
        for n in E.NAMES:
            m, b = E.load_point_cloud(n, layers[n])
            mc[n] = np.abs(m) if len(m) else m
            bl[n] = b
            ids[n] = E.load_id(n, layers[n])
        common = set(lang_floor)
        for n in E.NAMES:
            common &= set(ids[n].keys())
        common = sorted(common)
        idv = {n: np.array([ids[n][l] for l in common]) for n in E.NAMES}

        for a, b, _kind in PAIRS:
            u, p = E.mann_whitney_u(mc[a], mc[b])
            res[(a, b, "Max_Corr")].append(
                dict(round=k, p=float(p) if p is not None else None,
                     eff=E.cliffs_delta(mc[a], mc[b]),
                     la=layers[a], lb=layers[b]))
            u, p = E.mann_whitney_u(bl[a], bl[b])
            res[(a, b, "Best_Lag_Sec")].append(
                dict(round=k, p=float(p) if p is not None else None,
                     eff=E.cliffs_delta(bl[a], bl[b]),
                     la=layers[a], lb=layers[b]))
            t, p = E.welch_ttest(idv[a], idv[b])
            res[(a, b, "Intrinsic_Dim")].append(
                dict(round=k, p=float(p) if p is not None else None,
                     eff=E.cohen_d(idv[a], idv[b]),
                     la=layers[a], lb=layers[b]))
            # area: signed Max_Corr for the hull, as in manifold_area.csv
            pa = np.column_stack([E.load_point_cloud(a, layers[a])[0], bl[a]])
            pb = np.column_stack([E.load_point_cloud(b, layers[b])[0], bl[b]])
            d, lo, hi = E.bootstrap_area_diff(pa, pb, seed=k)
            res[(a, b, "Area(diff)")].append(
                dict(round=k, p=None, eff=float(d),
                     lo=float(lo), hi=float(hi),
                     sig=bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                     la=layers[a], lb=layers[b]))
    return res


def summarise(res, loops, n_tests):
    rows = []
    for (a, b, metric), recs in res.items():
        if metric == "Area(diff)":
            sig = [r for r in recs if r["sig"]]
            pos = sum(1 for r in sig if r["eff"] > 0)
            rows.append(dict(pair="%s-%s" % (a, b), metric=metric,
                             n=len(recs), sig05=len(sig), sigbonf=len(sig),
                             sign="%d+ / %d-" % (pos, len(sig) - pos) if sig else "-",
                             median_eff=float(np.median([r["eff"] for r in recs])),
                             all_same_sign=bool(len(sig) == len(recs) and (pos in (0, len(sig))))))
        else:
            ok = [r for r in recs if r["p"] is not None]
            sig = [r for r in ok if r["p"] < 0.05]
            bonf = [r for r in ok if r["p"] < 0.05 / n_tests]
            pos = sum(1 for r in sig if r["eff"] > 0)
            rows.append(dict(pair="%s-%s" % (a, b), metric=metric,
                             n=len(ok), sig05=len(sig), sigbonf=len(bonf),
                             sign="%d+ / %d-" % (pos, len(sig) - pos) if sig else "-",
                             median_eff=float(np.median([r["eff"] for r in ok])),
                             all_same_sign=bool(len(sig) == len(ok) and (pos in (0, len(sig))))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--loops", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results",
                                                "pair_summary.csv"))
    a = ap.parse_args()

    res = run(a.loops)
    n_tests = a.loops * len(PAIRS) * 3
    rows = summarise(res, a.loops, n_tests)

    print("Bonferroni threshold for %d tests: p < %.2e\n" % (n_tests, 0.05 / n_tests))
    hdr = "%-22s %-14s %5s %7s %8s %11s %10s %s"
    print(hdr % ("pair", "metric", "n", "p<.05", "Bonf", "sign", "median eff", "all same sign"))
    print("-" * 104)
    for r in rows:
        print(hdr % (r["pair"], r["metric"], r["n"], r["sig05"], r["sigbonf"],
                     r["sign"], "%.3f" % r["median_eff"],
                     "YES" if r["all_same_sign"] else "no"))

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % a.out)

    # The shipped table in scripts/ is the one the paper is built on. A re-run on a
    # trimmed cache set can differ; say so rather than letting the difference pass.
    shipped = os.path.join(HERE, "pair_summary.csv")
    if os.path.exists(shipped):
        import hashlib
        h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h(a.out) != h(shipped):
            print("[WARNING] this run differs from the shipped scripts/pair_summary.csv.")
            print("          The shipped table is authoritative for the paper; it was produced")
            print("          on the full per-node caches. 43_11 reads the shipped file.")
            return 4
        print("[OK] matches the shipped scripts/pair_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
