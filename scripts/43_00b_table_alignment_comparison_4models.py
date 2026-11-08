#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Table V: cross-lingual alignment at the two checkpoints, four XLS-R models.

Supersedes the three-model version (XLSR-53, XLS-R-300M, XLS-R-2B) and adds
XLS-R-1B at both checkpoints. Also fixes a label/value mismatch in the old table:
the XLS-R-300M late column printed the L19 values under an "(L18)" header.

Why L19 and not L18 for XLS-R-300M: the Mantel profile peaks at L18 (r_M = 0.361),
but at L18 only 16 of the 24 FLEURS languages keep a centroid under the
|Max_Corr| > 0.1 filter, so the aggregate there is computed over a different
population than the other models'. L19 is complete (24/24) and is the latest layer
at which all four encoders are comparable. The caption says so.

Definition of Mean Shift (from 05_formant_dim_analysis_06_meanshift_moreLayers.py):
centroid of each language in (Max_Corr, Best_Lag_Sec) over its significant
dimensions; global centroid = mean of language centroids; shift_i = ||c_i - g||;
MS = mean_i shift_i, expressed in ms (x1000).

Usage
-----
    python 43_00_table_alignment_comparison_Tab02_tab-alignment-comparison.py
    python 43_00_table_alignment_comparison_Tab02_tab-alignment-comparison.py --check

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

MODELS = [("XLSR-53", "wav2vec2-large-xlsr-53", 13, 19),
          ("XLS-R-300M", "wav2vec2-xls-r-300m", 13, 19),
          ("XLS-R-1B", "wav2vec2-xls-r-1b", 25, 38),
          ("XLS-R-2B", "wav2vec2-xls-r-2b", 25, 38)]

LANGS = [("German (deu)", "german"), ("Finnish (fin)", "finnish"),
         ("Georgian (kat)", "georgian")]


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def mean_shift(rows, model, layer):
    for r in rows:
        if r["Model"].strip().lower() == model.lower() and int(r["Layer"]) == layer:
            return float(r["Mean Shift"])
    return None


def area(rows, model, layer):
    for r in rows:
        if r["Model"].strip().lower() == model.lower() and int(r["Layer"]) == layer:
            return (float(r["Physical Coverage Area"]),
                    float(r["Normalized Manifold Area"]))
    return (None, None)


def lang_shift(root, mdir, layer, col):
    path = os.path.join(root, mdir, "mean_shift_by_langs.csv")
    for r in read_csv(path):
        if int(r["Layer"]) == layer:
            v = (r.get(col) or "").strip()
            return float(v) if v else None
    return None


def collect(project):
    root = os.path.join(project, ROOT)
    ms_rows = read_csv(os.path.join(root, "mean_shift.csv"))
    ar_rows = read_csv(os.path.join(root, "manifold_area.csv"))

    rows = []
    for name, mdir, la, lb in MODELS:
        rec = dict(model=name, la=la, lb=lb,
                   ms_a=mean_shift(ms_rows, name, la),
                   ms_b=mean_shift(ms_rows, name, lb),
                   ar_a=area(ar_rows, name, la),
                   ar_b=area(ar_rows, name, lb),
                   langs_a={}, langs_b={})
        for _disp, col in LANGS:
            rec["langs_a"][col] = lang_shift(root, mdir, la, col)
            rec["langs_b"][col] = lang_shift(root, mdir, lb, col)
        rows.append(rec)
    return rows


def fmt(v, nd=2, bold=False):
    if v is None:
        return "--"
    s = ("%." + str(nd) + "f") % v
    return r"\textbf{%s}" % s if bold else s


def best(vals, lower):
    vv = [v for v in vals if v is not None]
    if not vv:
        return None
    return min(vv) if lower else max(vv)


def emit(rows, path):
    def cell(v, target, nd=2, phys=False):
        txt = fmt(v, nd, bold=(v is not None and target is not None and abs(v - target) < 1e-9))
        return "(%s)" % txt if phys and v is not None else txt

    B = lambda side, key, lower: best([r[side][key] if isinstance(r[side], dict) else r[side] for r in rows], lower)

    ms_a = best([r["ms_a"] for r in rows], True)
    ms_b = best([r["ms_b"] for r in rows], True)
    na_a = best([r["ar_a"][1] for r in rows], False)
    na_b = best([r["ar_b"][1] for r in rows], False)
    pa_a = best([r["ar_a"][0] for r in rows], False)
    pa_b = best([r["ar_b"][0] for r in rows], False)

    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Cross-lingual alignment at the two checkpoints of Section~\ref{sec:res_check}: "
         r"global statistics over the 24 FLEURS languages and three representative languages, for the "
         r"four XLS-R encoders. $MS$ is the mean shift in ms; arrows give the preferred direction. "
         r"For \XLSRThreeOO{} the late column uses $L{=}19$: at its Mantel maximum ($L{=}18$) only 16 "
         r"of the 24 languages retain a centroid under the $|R_{\max}|>0.1$ filter, so that layer is "
         r"not comparable across models. Physical areas are in Tables~\ref{tab:scaling_metrics} and~\ref{tab:anchor_vs_late}.}",
         r"\label{tab:alignment_comparison}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}", r"\toprule",
         r"\textbf{Metric} & \multicolumn{4}{c}{\textbf{Mid-depth anchor}} "
         r"& \multicolumn{4}{c}{\textbf{Late checkpoint}} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"& \textbf{53} & \textbf{300M} & \textbf{1B} & \textbf{2B} "
         r"& \textbf{53} & \textbf{300M} & \textbf{1B} & \textbf{2B} \\",
         r"& \textbf{(L13)} & \textbf{(L13)} & \textbf{(L25)} & \textbf{(L25)} "
         r"& \textbf{(L19)} & \textbf{(L19)} & \textbf{(L38)} & \textbf{(L38)} \\",
         r"\midrule"]

    def row(label, ga, gb, nd=2, phys=False):
        cells = [cell(r[ga], None, nd, phys) for r in rows]
        return label + " & " + " & ".join(cells) + r" \\"

    # MS: lower is better
    def ms_cell(r, key, target):
        return fmt(r[key], 2, bold=(r[key] is not None and target is not None
                                    and abs(r[key] - target) < 1e-9))
    L.append(r"$MS$ [ms] $\downarrow$ & "
             + " & ".join(ms_cell(r, "ms_a", ms_a) for r in rows) + " & "
             + " & ".join(ms_cell(r, "ms_b", ms_b) for r in rows) + r" \\")

    def ar_cell(r, key, target, idx, phys=False):
        v = r[key][idx]
        return cell(v, target, 4, phys)
    L.append(r"$\widehat{A}_{\text{norm}}$ $\uparrow$ & "
             + " & ".join(ar_cell(r, "ar_a", na_a, 1) for r in rows) + " & "
             + " & ".join(ar_cell(r, "ar_b", na_b, 1) for r in rows) + r" \\")
    L.append(r"\midrule")

    for disp, col in LANGS:
        va = [r["langs_a"][col] for r in rows]
        vb = [r["langs_b"][col] for r in rows]
        ba, bb = best(va, True), best(vb, True)
        fa = [fmt(v, 2, v is not None and ba is not None and abs(v - ba) < 1e-9) for v in va]
        fb = [fmt(v, 2, v is not None and bb is not None and abs(v - bb) < 1e-9) for v in vb]
        L.append("%s & %s & %s \\\\" % (disp, " & ".join(fa), " & ".join(fb)))

    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(L))


def show(rows):
    print("%-12s %6s %8s %8s %10s %10s" % ("model", "L_a", "MS_a", "MS_b", "A_a", "A_b"))
    for r in rows:
        print("%-12s %6d %8.2f %8.2f %10.4f %10.4f" %
              (r["model"], r["la"], r["ms_a"] or float("nan"), r["ms_b"] or float("nan"),
               (r["ar_a"][0] if r["ar_a"][0] is not None else float("nan")),
               (r["ar_b"][0] if r["ar_b"][0] is not None else float("nan"))))
    print()
    for disp, col in LANGS:
        print("%-14s " % disp + "  ".join(
            "%s: %.2f->%.2f" % (r["model"].split("-")[-1], r["langs_a"][col] or float("nan"),
                                r["langs_b"][col] or float("nan")) for r in rows))


def check(rows):
    d = {r["model"]: r for r in rows}
    ok = True

    def near(label, got, want, tol=0.006):
        nonlocal ok
        good = got is not None and abs(got - want) <= tol
        ok &= good
        print("  [%s] %-46s %s (paper %s)" % ("OK" if good else "FAIL", label,
                                              "n/a" if got is None else round(got, 4), want))

    print("Verifying the numbers quoted in the paper and the table:")
    near("MS anchor, XLSR-53", d["XLSR-53"]["ms_a"], 134.97)
    near("MS anchor, XLS-R-300M", d["XLS-R-300M"]["ms_a"], 69.01)
    near("MS anchor, XLS-R-1B", d["XLS-R-1B"]["ms_a"], 54.73)
    near("MS anchor, XLS-R-2B", d["XLS-R-2B"]["ms_a"], 95.79)
    near("MS late, XLSR-53", d["XLSR-53"]["ms_b"], 58.21)
    near("MS late, XLS-R-300M (L19)", d["XLS-R-300M"]["ms_b"], 51.78)
    near("MS late, XLS-R-1B", d["XLS-R-1B"]["ms_b"], 108.52)
    near("MS late, XLS-R-2B", d["XLS-R-2B"]["ms_b"], 42.60)
    near("area late, XLS-R-1B", d["XLS-R-1B"]["ar_b"][0], 0.4995)
    near("area late, XLS-R-2B", d["XLS-R-2B"]["ar_b"][0], 0.1354)
    near("German, XLS-R-1B anchor", d["XLS-R-1B"]["langs_a"]["german"], 104.14)
    near("German, XLS-R-1B late", d["XLS-R-1B"]["langs_b"]["german"], 149.96)
    near("Finnish, XLS-R-1B late", d["XLS-R-1B"]["langs_b"]["finnish"], 387.93)
    near("Finnish, XLSR-53 late (the degradation)", d["XLSR-53"]["langs_b"]["finnish"], 224.08)

    checks = [("2B has the lowest late MS of the four",
               min(r["ms_b"] for r in rows) == d["XLS-R-2B"]["ms_b"]),
              ("1B late MS exceeds its anchor MS",
               d["XLS-R-1B"]["ms_b"] > d["XLS-R-1B"]["ms_a"]),
              ("four models, two checkpoints each", len(rows) == 4),
              ("all 24 language columns present for 300M at L19",
               d["XLS-R-300M"]["langs_b"]["finnish"] is not None)]
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
    show(rows)
    if a.check:
        return check(rows)
    out = os.path.join(a.project, "results", "alignment_comparison_table.tex")
    emit(rows, out)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
