#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Downstream check: does any geometric quantity order recognition performance?

Reproduces Table `tab:downstream_cer` of geometry_revised.tex together with the
three statements the text makes about it.

Protocol behind the numbers (see asr-ctc/run_pipeline.py):
  frozen SSL features, one hour per language, per-layer hidden states saved as
  float16; a shallow CTC head (learnable softmax weights over 8 evenly spaced layers -> linear ->
  Conv1d(k=3, s=2) -> 2 Transformer layers, d_model 256, 8 heads) is trained
  per language on the monolingual track for 30 epochs with seed 42 and greedy
  decoding; CER is scored on the FLEURS test split. All seven encoders go
  through the identical recipe.

Inputs
------
~/Documents/asr-ctc/results_2026-10-28_18-28-43/asr.csv   model,lang,metric,value,...
<project>/re-organized/<model>/ID_result/                  intrinsic dimension
<project>/scripts/../tables/cross_family_metrics.csv       (optional reference)

Outputs
-------
results/downstream_cer_table.tex
results/downstream_cer_metrics.csv
stdout

Usage
-----
    python 43_07_table_downstream_cer_tab-downstream_cer.py
    python 43_07_table_downstream_cer_tab-downstream_cer.py --check

Standard library only.
"""

import argparse
import collections
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

# display name, asr.csv key, re-organized directory, layers (incl. input)
MODELS = [("XLSR-53", "xlsr53", "wav2vec2-large-xlsr-53", 25),
          ("XLS-R-300M", "xlsr300m", "wav2vec2-xls-r-300m", 25),
          ("XLS-R-1B", "xlsr1b", "wav2vec2-xls-r-1b", 49),
          ("XLS-R-2B", "xlsr2b", "wav2vec2-xls-r-2b", 49),
          ("MMS-300M", "mms300m", "mms-300m", 25),
          ("MMS-1B", "mms1b", "mms-1b", 49),
          ("mHuBERT-147", "mhubert147", "mHuBERT-147", 13)]

DEPTHS = (0.35, 0.50, 0.65)          # the three depths of Table `cross_family`


def read_asr(path):
    """{lang: {model_key: CER}}"""
    tab = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["metric"] == "CER":
            tab[r["lang"]][r["model"]] = float(r["value"])
    return tab


def id_band(root, model_dir, n_layers):
    """Mean intrinsic dimension over the 24 FLEURS languages at three depths."""
    d = os.path.join(root, model_dir, "ID_result")
    vals = []
    for rel in DEPTHS:
        L = int(round(rel * (n_layers - 1)))
        p = os.path.join(d, "ID_Analysis_L%d_Fixed.csv" % L)
        if not os.path.exists(p):
            return None
        v = [float(r["Intrinsic_Dim"])
             for r in csv.DictReader(open(p, encoding="utf-8"))
             if r["Language"].endswith("_fleurs")]
        vals.append(sum(v) / len(v))
    return sum(vals) / len(vals)


def mean(xs):
    return sum(xs) / len(xs)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma, mb = mean(ra), mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = (sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb)) ** 0.5
    return num / den if den else float("nan")


def collect(project):
    asr = os.path.join(os.path.expanduser("~"), "Documents", "asr-ctc",
                       "results_2026-10-28_18-28-43", "asr.csv")
    tab = read_asr(asr)
    root = os.path.join(project, "re-organized")

    rows = []
    for disp, key, mdir, n in MODELS:
        cers = [tab[L][key] for L in tab if key in tab[L]]
        rows.append(dict(model=disp, key=key, n=len(cers),
                         cer=mean(cers),
                         id=id_band(root, mdir, n)))

    best = collections.Counter()
    worst = collections.Counter()
    spread = {}
    for L, d in tab.items():
        items = sorted(d.items(), key=lambda kv: kv[1])
        best[items[0][0]] += 1
        worst[items[-1][0]] += 1
        spread[L] = items[-1][1] - items[0][1]
    for r in rows:
        r["best"] = best.get(r["key"], 0)
        r["worst"] = worst.get(r["key"], 0)
        r["spread"] = None

    shared = [L for L in tab if all(k in tab[L] for _d, k, _m, _n in MODELS)]
    pairs = {}
    for a, b in [("xlsr2b", "xlsr1b"), ("xlsr2b", "xlsr300m"),
                 ("xlsr1b", "xlsr300m"), ("xlsr53", "xlsr300m"),
                 ("mms1b", "xlsr1b")]:
        diffs = [tab[L][a] - tab[L][b] for L in shared]
        pairs[(a, b)] = dict(delta=mean(diffs),
                             a_worse=sum(1 for x in diffs if x > 0),
                             n=len(diffs))
    rho = spearman([r["id"] for r in rows], [r["cer"] for r in rows])
    sp = sorted(spread.items(), key=lambda kv: -kv[1])
    return dict(rows=rows, pairs=pairs, rho=rho, shared=len(shared),
                spread_hi=sp[:4], spread_lo=sp[-4:][::-1])


def write_tex(res, path):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{The three geometric quantities against recognition performance. "
        r"$\widehat{id}$ is the mid-depth band mean of Table~\ref{tab:cross_family}; "
        r"CER is the mean over the 24 FLEURS languages of a frozen-feature CTC probe "
        r"(one hour of features per language, 30 epochs, greedy decoding). ``Best'' and "
        r"``worst'' count the languages in which the encoder places first or last among "
        r"the seven. Lower CER is better; higher $\widehat{id}$ is not.}"
        r"\label{tab:downstream_cer}",
        r"\small",
        r"\begin{adjustbox}{width=\columnwidth,center}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Encoder} & \textbf{$\widehat{id}$} & \textbf{CER (\%)} & "
        r"\textbf{Best} & \textbf{Worst} & \textbf{Languages} \\",
        r"\midrule",
    ]
    for r in res["rows"]:
        lines.append("%s & %.2f & %.2f & %d & %d & %d \\\\" %
                     (r["model"], r["id"], r["cer"], r["best"], r["worst"], r["n"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def print_all(res):
    print("%-12s %7s %9s %6s %6s %6s" % ("Encoder", "ID", "CER", "n", "best", "worst"))
    for r in res["rows"]:
        print("%-12s %7.2f %9.2f %6d %6d %6d" %
              (r["model"], r["id"], r["cer"], r["n"], r["best"], r["worst"]))
    print("\nSpearman(ID, CER) over 7 encoders: %.3f" % res["rho"])
    print("shared languages with all seven encoders: %d" % res["shared"])
    for (a, b), v in res["pairs"].items():
        print("  %s - %s: dmean=%+.2f, %s worse in %d/%d languages" %
              (a, b, v["delta"], a, v["a_worse"], v["n"]))
    print("largest spread:", ", ".join("%s %.1f" % (L, s) for L, s in res["spread_hi"]))
    print("smallest spread:", ", ".join("%s %.1f" % (L, s) for L, s in res["spread_lo"]))


def check(res):
    d = {r["key"]: r for r in res["rows"]}
    ok = True

    def near(label, got, want, tol=0.006):
        nonlocal ok
        good = got is not None and abs(got - want) <= tol
        ok &= good
        print("  [%s] %-44s %s (paper %s)" % ("OK" if good else "FAIL", label,
                                              "n/a" if got is None else round(got, 3), want))

    print("Verifying the numbers quoted in the paper:")
    for k, want in [("xlsr53", 37.78), ("xlsr300m", 31.84), ("xlsr1b", 30.72),
                    ("xlsr2b", 32.33), ("mms300m", 29.45), ("mms1b", 23.21),
                    ("mhubert147", 26.94)]:
        near("mean CER, %s" % k, d[k]["cer"], want, 0.01)
    near("ID band, XLS-R-2B (the highest)", d["xlsr2b"]["id"], 17.04, 0.01)
    near("ID band, mHuBERT-147 (the lowest)", d["mhubert147"]["id"], 11.72, 0.01)
    near("Spearman(ID, CER)", res["rho"], -0.04, 0.03)

    p = res["pairs"]
    near("2B - 1B mean CER", p[("xlsr2b", "xlsr1b")]["delta"], 1.52, 0.02)
    near("2B - 300M mean CER", p[("xlsr2b", "xlsr300m")]["delta"], 0.17, 0.02)
    near("1B - 300M mean CER", p[("xlsr1b", "xlsr300m")]["delta"], -1.34, 0.02)
    near("53 - 300M mean CER", p[("xlsr53", "xlsr300m")]["delta"], 7.17, 0.02)
    near("MMS-1B - XLS-R-1B mean CER", p[("mms1b", "xlsr1b")]["delta"], -7.21, 0.02)

    checks = [
        ("2B worse than 1B in only 4 of 23 languages", p[("xlsr2b", "xlsr1b")]["a_worse"] == 4),
        ("53 worse than 300M in 22 of 23 languages", p[("xlsr53", "xlsr300m")]["a_worse"] == 22),
        ("MMS-1B worse than XLS-R-1B in 0 of 23", p[("mms1b", "xlsr1b")]["a_worse"] == 0),
        ("MMS-1B is best in 23 languages", d["mms1b"]["best"] == 23),
        ("XLSR-53 is worst in 19 languages", d["xlsr53"]["worst"] == 19),
        ("shared language count is 23", res["shared"] == 23),
    ]
    for label, good in checks:
        ok &= good
        print("  [%s] %s" % ("OK" if good else "FAIL", label))

    print("\n%s" % ("ALL NUMBERS MATCH THE PAPER" if ok else "MISMATCH -- do not compile"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    project = _project_root()
    ap.add_argument("--project", default=project)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = collect(a.project)
    print_all(res)
    if a.check:
        return check(res)
    out = os.path.join(a.project, "results", "downstream_cer_table.tex")
    write_tex(res, out)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
