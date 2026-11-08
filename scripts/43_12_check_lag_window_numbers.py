#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute the two lag-window numbers quoted in Section VI-A and in R1.5.

The sentence is:

    The lag search runs to +-2 s ... The data use the range---some dimensions sit at the
    limit---and a 200 ms bound would drop 9.9% of the dimensions retained at |r| > 0.1.

Both halves are checked here against the per-dimension formant table, so the number in
the paper is never one that only existed in a previous draft.

    python 43_12_check_lag_window_numbers.py

Exits non-zero if either quoted value no longer holds. `--verbose` prints the per-model
breakdown.
"""

import argparse
import collections
import csv
import glob
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

ROOT = os.path.join(_project_root(), "re-organized")
QUOTED_PCT = 9.9      # Section VI-A
QUOTED_LIMIT_S = 2.0  # MAX_LAG_SEC in 02_02_formant_dim_analysis_02_praat_sync.py


def load(root=ROOT):
    """One row per (model, language, layer, dimension), max |r| across F1-F3."""
    files = sorted(glob.glob(os.path.join(root, "*", "formant_dim_analysis_cache",
                                          "Formant_Lag_Analysis_Results_moreLayers.csv")))
    if not files:
        raise SystemExit("[ERROR] no Formant_Lag_Analysis_Results_moreLayers.csv under %s" % root)
    best = collections.defaultdict(lambda: (0.0, 0.0))
    for f in files:
        model = os.path.basename(os.path.dirname(os.path.dirname(f)))
        with open(f, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                key = (model, r["Language"], int(r["Layer"]), int(r["Dimension"]))
                c, t = abs(float(r["Max_Corr"])), float(r["Best_Lag_Sec"])
                if c > best[key][0]:
                    best[key] = (c, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    best = load()
    corr = np.array([v[0] for v in best.values()])
    lag = np.array([v[1] for v in best.values()])
    models = np.array([k[0] for k in best])

    # (1) the search really covers +-2 s: values pinned at the bound
    at_limit = int((np.abs(lag) >= QUOTED_LIMIT_S - 1e-9).sum())
    # (2) the quoted percentage, at the paper's retention rule |R_max| > 0.1
    keep = corr > 0.1
    pct = 100.0 * float((np.abs(lag[keep]) > 0.2).mean())

    print("dimensions (model x language x layer), max over F1-F3 : %d" % len(best))
    print("retained at |r| > 0.1                                : %d" % int(keep.sum()))
    print("of those, |tau| > 200 ms                             : %.1f%%  (paper says %.1f%%)"
          % (pct, QUOTED_PCT))
    print("dimensions sitting at the +-%.0f s bound              : %d"
          % (QUOTED_LIMIT_S, at_limit))

    if a.verbose:
        print()
        for m in sorted(set(models)):
            sel = keep & (models == m)
            if sel.sum():
                print("  %-28s n=%5d  |tau|>200ms %5.1f%%"
                      % (m, int(sel.sum()), 100 * float((np.abs(lag[sel]) > 0.2).mean())))

    ok = abs(pct - QUOTED_PCT) < 0.15 and at_limit > 0
    print("\n%s" % ("OK: both quoted values reproduce." if ok else "MISMATCH: the text needs updating."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
