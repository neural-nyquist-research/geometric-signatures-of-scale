#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Anchor vs. alignment checkpoint: the same four encoders at two layer choices.

Reproduces Table `tab:anchor_vs_late` of geometry_revised.tex.

Two layer sets, both defined in the paper:
  anchor        L13 / L13 / L25 / L25   -- round(0.5*(n-1))+1, a convention
  checkpoint    L19 / L18 / L38 / L38   -- late Mantel maximum per model
                                          (results/mantel_checkpoints_metrics.csv)

Reported per encoder and layer set: mean intrinsic dimension over the 24 FLEURS
languages, physical convex-hull area, and normalised area. The inferential results
quoted in the text come from the statistical driver:

    python 02_07_formant_dim_statistical_test_comprehensive.py --late
    python 02_07_formant_dim_statistical_test_comprehensive.py --layers 19,19,38,38

(the second substitutes matched relative depth 0.792 for the 300M argmax, and
reaches the same verdict).

Usage
-----
    python 43_08_table_anchor_vs_late_tab-anchor_vs_late.py
    python 43_08_table_anchor_vs_late_tab-anchor_vs_late.py --check

Standard library only.
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

ROOT = "re-organized"

MODELS = [("XLSR-53", "wav2vec2-large-xlsr-53"),
          ("XLS-R-300M", "wav2vec2-xls-r-300m"),
          ("XLS-R-1B", "wav2vec2-xls-r-1b"),
          ("XLS-R-2B", "wav2vec2-xls-r-2b")]

ANCHOR = {"XLSR-53": 13, "XLS-R-300M": 13, "XLS-R-1B": 25, "XLS-R-2B": 25}
LATE = {"XLSR-53": 19, "XLS-R-300M": 18, "XLS-R-1B": 38, "XLS-R-2B": 38}


def area_table(root):
    out = {}
    path = os.path.join(root, "manifold_area.csv")
    names = [m for m, _ in MODELS]
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = next((n for n in names if n.lower() == row["Model"].strip().lower()), None)
            if m:
                out[(m, int(row["Layer"]))] = (float(row["Physical Coverage Area"]),
                                               float(row["Normalized Manifold Area"]))
    return out


def id_mean(root, mdir, layer):
    path = os.path.join(root, mdir, "ID_result", "ID_Analysis_L%d_Fixed.csv" % layer)
    vals = [float(r["Intrinsic_Dim"]) for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["Language"].endswith("_fleurs")]
    return sum(vals) / len(vals)


def collect(project):
    root = os.path.join(project, ROOT)
    areas = area_table(root)
    rows = []
    for name, mdir in MODELS:
        a, b = ANCHOR[name], LATE[name]
        rows.append(dict(
            model=name, anchor=a, late=b,
            id_a=id_mean(root, mdir, a), id_b=id_mean(root, mdir, b),
            ar_a=areas[(name, a)][0], ar_b=areas[(name, b)][0],
            nm_a=areas[(name, a)][1], nm_b=areas[(name, b)][1]))
    return rows


def write_tex(rows, path):
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{The same four encoders at the mid-depth anchor (L13/L13/L25/L25) and at "
         r"the late alignment checkpoint (L19/L18/L38/L38, the per-model Mantel maximum). "
         r"$\widehat{id}$ is the mean over the 24 FLEURS languages; area is the physical "
         r"convex hull of the $(Max\_Corr, \Delta t)$ cloud, $A_{\text{norm}}$ its size "
         r"relative to the full plane. Every quantity reverses its ordering between the two "
         r"layer sets.}",
         r"\label{tab:anchor_vs_late}", r"\small",
         r"\begin{adjustbox}{width=\columnwidth,center}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{\textbf{Mid-depth anchor}} "
         r"& \multicolumn{3}{c}{\textbf{Alignment checkpoint}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Encoder} & \textbf{Layer} & $\widehat{id}$ & \textbf{Area} & \textbf{Layer} "
         r"& $\widehat{id}$ & \textbf{Area} \\", r"\midrule"]
    for r in rows:
        L.append("\\mbox{%s} & %d & %.2f & %.4f & %d & %.2f & %.4f \\\\" %
                 (r["model"], r["anchor"], r["id_a"], r["ar_a"],
                  r["late"], r["id_b"], r["ar_b"]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(L))


def print_all(rows):
    print("%-12s %5s %6s %8s %8s | %5s %6s %8s %8s" %
          ("Encoder", "L_a", "ID_a", "Area_a", "Norm_a", "L_b", "ID_b", "Area_b", "Norm_b"))
    for r in rows:
        print("%-12s %5d %6.2f %8.4f %8.4f | %5d %6.2f %8.4f %8.4f" %
              (r["model"], r["anchor"], r["id_a"], r["ar_a"], r["nm_a"],
               r["late"], r["id_b"], r["ar_b"], r["nm_b"]))
    a1 = next(r for r in rows if r["model"] == "XLS-R-1B")
    a2 = next(r for r in rows if r["model"] == "XLS-R-2B")
    print("\n1B vs 2B at the anchor:  area %.4f vs %.4f  (2B larger by %.1fx);  ID %.2f vs %.2f"
          % (a1["ar_a"], a2["ar_a"], a2["ar_a"] / a1["ar_a"], a1["id_a"], a2["id_a"]))
    print("1B vs 2B at checkpoint: area %.4f vs %.4f  (1B larger by %.1fx);  ID %.2f vs %.2f"
          % (a1["ar_b"], a2["ar_b"], a1["ar_b"] / a2["ar_b"], a1["id_b"], a2["id_b"]))


def check(rows):
    d = {r["model"]: r for r in rows}
    ok = True

    def near(label, got, want, tol=0.006):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print("  [%s] %-40s %8.4f (paper %s)" % ("OK" if good else "FAIL", label, got, want))

    print("Verifying the 24 numbers of the table:")
    for m, ida, ara, nm_a, idb, arb, nm_b in [
            ("XLSR-53", 15.55, 0.4154, 0.0775, 12.90, 0.1309, 0.0213),
            ("XLS-R-300M", 15.61, 0.0442, 0.0073, 13.62, 0.2976, 0.0571),
            ("XLS-R-1B", 16.87, 0.2210, 0.0313, 16.45, 0.4995, 0.0750),
            ("XLS-R-2B", 17.33, 0.6010, 0.0875, 15.12, 0.1354, 0.0220)]:
        near("%s anchor ID" % m, d[m]["id_a"], ida, 0.01)
        near("%s anchor area" % m, d[m]["ar_a"], ara)
        near("%s anchor norm" % m, d[m]["nm_a"], nm_a)
        near("%s late ID" % m, d[m]["id_b"], idb, 0.01)
        near("%s late area" % m, d[m]["ar_b"], arb)
        near("%s late norm" % m, d[m]["nm_b"], nm_b)

    a1, a2 = d["XLS-R-1B"], d["XLS-R-2B"]
    checks = [
        ("2B area exceeds 1B at the anchor", a2["ar_a"] > a1["ar_a"]),
        ("1B area exceeds 2B at the checkpoint", a1["ar_b"] > a2["ar_b"]),
        ("2B ID exceeds 1B at the anchor", a2["id_a"] > a1["id_a"]),
        ("1B ID exceeds 2B at the checkpoint", a1["id_b"] > a2["id_b"]),
        ("every ID falls from anchor to checkpoint",
         all(r["id_b"] < r["id_a"] for r in rows)),
        ("4 models present", len(rows) == 4),
    ]
    print("\nStructural claims:")
    for label, good in checks:
        ok &= good
        print("  [%s] %s" % ("OK" if good else "FAIL", label))

    print("\n%s" % ("ALL NUMBERS MATCH THE PAPER" if ok else "MISMATCH -- do not compile"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--project", default=_project_root())
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = collect(a.project)
    print_all(rows)
    if a.check:
        return check(rows)
    out = os.path.join(a.project, "results", "anchor_vs_late_table.tex")
    write_tex(rows, out)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
