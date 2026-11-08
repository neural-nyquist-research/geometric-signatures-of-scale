#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table: how far the capacity contrasts replicate across the twelve rounds.

Reads pair_summary.csv (written by 43_10_pairwise_capacity_pairs.py) and emits
results/pair_contrasts_table.tex.

Shape: one row per metric, one column per capacity contrast, cell = how many of
the twelve relative-depth rounds reach p < 0.05 together with the sign split among
those rounds. The 1B-2B column is the contrast the paper had been reporting; the
two 300M columns are what it was missing.

XLSR-53 is deliberately absent: it is trained on 53 languages rather than 128, so
a contrast against it measures data, not capacity.

Usage
-----
    python 43_11_table_pair_contrasts.py
    python 43_11_table_pair_contrasts.py --check
"""

import argparse
import csv
import os

def _project_root():
    """The directory holding re-organized/. Current directory, then the script's, then its parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = _project_root()

PAIRS = [("XLS-R-300M-XLS-R-1B", "300M $\\rightarrow$ 1B"),
         ("XLS-R-300M-XLS-R-2B", "300M $\\rightarrow$ 2B"),
         ("XLS-R-1B-XLS-R-2B", "1B $\\rightarrow$ 2B")]

ROWS = [("Intrinsic_Dim", "$\\widehat{id}$"),
        ("Max_Corr", "$|R_{\\max}|$"),
        ("Best_Lag_Sec", "Best\\,lag"),
        ("Area(diff)", "Hull area")]

# what the paper quotes in Section IV-B, checked by --check
EXPECT = {
    ("XLS-R-300M-XLS-R-1B", "Intrinsic_Dim"): (10, 0, 10),
    ("XLS-R-300M-XLS-R-2B", "Intrinsic_Dim"): (11, 0, 11),
    ("XLS-R-1B-XLS-R-2B", "Intrinsic_Dim"): (10, 6, 4),
}


def load(path):
    return {(r["pair"], r["metric"]): r for r in csv.DictReader(open(path, encoding="utf-8"))}


def emit(tab, path):
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{How far each capacity contrast replicates. Cell entries give the number of "
         r"the twelve relative-depth rounds reaching $p<0.05$ and, among those, the sign split "
         r"($a{+}/{:}b{-}$ for the first against the second model). Effects are per-node for "
         r"$|R_{\max}|$ and lag, per-language for $\widehat{id}$, and bootstrap for hull area. "
         r"\XLSRFiveThree{} is excluded as a capacity contrast: it is trained on 53 languages "
         r"rather than 128, so a contrast against it measures data, not scale.}",
         r"\label{tab:pair_contrasts}", r"\small",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}", r"\toprule",
         r"\textbf{Metric} & " + " & ".join(h for _k, h in PAIRS) + r" \\", r"\midrule"]

    for key, label in ROWS:
        cells = []
        for pk, _h in PAIRS:
            r = tab[(pk, key)]
            cells.append("%s/12 (%s)" % (r["sig05"], r["sign"].replace(" ", "")))
        L.append("%s & %s \\\\" % (label, " & ".join(cells)))

    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


def check(tab):
    ok = True
    for (pk, key), (sig, pos, neg) in EXPECT.items():
        r = tab[(pk, key)]
        got = (int(r["sig05"]),) + tuple(int(x) for x in r["sign"].replace(" ", "").split("/")[0:1][0].split("+")[:1] or [0])
        sign = r["sign"].replace(" ", "")
        p = int(sign.split("/")[0].split("+")[0])
        n = int(sign.split("/")[1].rstrip("-"))
        good = int(r["sig05"]) == sig and p == pos and n == neg
        ok &= good
        print("  [%s] %-22s %-13s sig=%s sign=%s (paper %d, %d+/%d-)"
              % ("OK" if good else "FAIL", pk, key, r["sig05"], sign, sig, pos, neg))
    print("\n%s" % ("ALL NUMBERS MATCH THE PAPER" if ok else "MISMATCH -- do not compile"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--csv", default=os.path.join(HERE, "pair_summary.csv"))
    ap.add_argument("--out", default=os.path.join(PROJECT, "results",
                                                 "pair_contrasts_table.tex"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        print("[ERROR] missing %s -- run 43_10_pairwise_capacity_pairs.py first" % a.csv)
        return 2
    tab = load(a.csv)
    if a.check:
        return check(tab)
    emit(tab, a.out)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
