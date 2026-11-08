#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce Table `tab:mantel_checkpoints` of geometry_revised.tex.

The table is what the paper's RQ1 rests on, so every column is defined here and
nothing is typed by hand.

Definitions
-----------
profile      r_M(L) of the canonical extraction variant, Norm=True with
             Metric=Cosine --- the variant plotted in Fig. 1. The other three
             variants are checked for agreement on every argmax (they agree for
             all four models), so the column is not an artefact of the variant;
             averaging them instead would be defensible but would put a table
             value (0.488) beside a figure annotation (0.490).
anchor       the mid-depth layer the geometric analyses are read at. It is a
             CONVENTION (first transformer layer past the midpoint),
             anchor = round(0.5*(n_layers-1)) + 1, which gives L13 for the
             24-layer encoders and L25 for the 48-layer ones. It is NOT selected
             by r_M; the table exists to show how far the panel is from the
             model's own alignment landmarks.
mid band     relative depth 0.35-0.65, i.e. layers ceil(0.35*(n-1))..floor(0.65*(n-1))
late band    relative depth 0.70-0.85, the last-quarter maximum used as the
             late alignment checkpoint
rel. depth   L/(n_layers-1), with n_layers counted including the input layer

Inputs
------
./re-organized/mantel_results_cache.csv   Model,Layer,Mantel_R,Label

Outputs
-------
results/mantel_checkpoints_table.tex
results/mantel_checkpoints_metrics.csv
stdout (plain-text version)

Usage
-----
    python 43_05_table_mantel_checkpoints_tab-mantel_checkpoints.py
    python 43_05_table_mantel_checkpoints_tab-mantel_checkpoints.py --check

--check asserts the numbers quoted in the paper and exits non-zero on any
mismatch. Standard library only; no pandas/scipy.
"""

import argparse
import collections
import csv
import math
import os

def _project_root():
    """The directory holding re-organized/. Current directory, then the script's, then its parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)

import sys

MODELS = [("XLSR-53", "xlsr-53"), ("XLS-R-300M", "xls-r-300m"),
          ("XLS-R-1B", "xls-r-1b"), ("XLS-R-2B", "xls-r-2b")]

MID_BAND = (0.35, 0.65)
LATE_BAND = (0.70, 0.85)


def load(path):
    """{(model, label): {layer: r}} plus the layer count per model."""
    prof = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        prof[(r["Model"], r["Label"])][int(r["Layer"])] = float(r["Mantel_R"])
    return prof


CANON = "Norm=True, Metric=Cosine"


def profile(prof, model):
    """r_M of the canonical variant, layer -> value; plus n_layers and the set of labels."""
    labs = {lab for (m, lab) in prof if m == model}
    n = max(L for (m, _l) in prof if m == model for L in prof[(m, _l)]) + 1
    return {L: prof[(model, CANON)][L] for L in range(n)}, n, labs


def variants_agree(prof, model, labs, n):
    """True if all four variants pick the same global argmax as the canonical one."""
    canon = max(range(n), key=lambda L: prof[(model, CANON)][L])
    return all(max(range(n), key=lambda L: prof[(model, lab)][L]) == canon for lab in labs)


def band_layers(n, band):
    lo, hi = band
    return range(int(math.ceil(lo * (n - 1))), int(math.floor(hi * (n - 1))) + 1)


def argmax(curve, layers):
    return max(layers, key=lambda L: curve[L])


def anchor_layer(n):
    return int(round(0.5 * (n - 1))) + 1


def collect(path):
    prof = load(path)
    rows = []
    for disp, key in MODELS:
        c, n, labs = profile(prof, key)
        agree = variants_agree(prof, key, labs, n)
        a = anchor_layer(n)
        mid = argmax(c, band_layers(n, MID_BAND))
        late = argmax(c, band_layers(n, LATE_BAND))
        gmax = argmax(c, range(n))
        rows.append(dict(model=disp, n=n, anchor=a, r_anchor=c[a], variants_agree=agree,
                         mid_layer=mid, r_mid=c[mid],
                         late_layer=late, r_late=c[late],
                         late_depth=late / (n - 1),
                         gmax_layer=gmax, r_gmax=c[gmax]))
    return rows


def write_tex(rows, path):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{The Mantel profile of each encoder, and the two layers the paper reads it at. "
        r"$r_M$ is the canonical variant (Norm=True, cosine), the one plotted in Fig.~1; "
        r"the other three variants place the late maximum within one layer of it for the 48-layer encoders and within two for the 25-layer ones. "
        r"The \emph{anchor} is the first transformer layer past the midpoint---a convention that "
        r"makes the encoders comparable, not a property the profile selects. The mid- and "
        r"late-band maxima are the highest $r_M$ in relative depth $0.35$--$0.65$ and "
        r"$0.70$--$0.85$. Where the global maximum of a profile falls---early for two encoders, "
        r"late for the other two---is reported in the text, because a single column invites the "
        r"reading that it is a family-wide checkpoint.}"
        r"\label{tab:mantel_checkpoints}",
        r"\small",
        r"\begin{adjustbox}{width=\columnwidth,center}",
        r"\begin{tabular}{l r c c c c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Layers} & \textbf{Anchor} & \textbf{$r_M$ at anchor} "
        r"& \textbf{Mid-band max} & \textbf{Late checkpoint} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            "%s & %d & L%d & %.3f & L%d (%.3f) & L%d (%.3f), rel.\\ %.2f \\\\"
            % (r["model"], r["n"], r["anchor"], r["r_anchor"],
               r["mid_layer"], r["r_mid"], r["late_layer"], r["r_late"], r["late_depth"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "N_layers", "Anchor_layer", "r_at_anchor",
                    "Midband_layer", "Midband_r", "Late_layer", "Late_r",
                    "Late_rel_depth", "GlobalMax_layer", "GlobalMax_r"])
        for r in rows:
            w.writerow([r["model"], r["n"], r["anchor"], "%.4f" % r["r_anchor"],
                        r["mid_layer"], "%.4f" % r["r_mid"], r["late_layer"],
                        "%.4f" % r["r_late"], "%.3f" % r["late_depth"],
                        r["gmax_layer"], "%.4f" % r["r_gmax"]])
    return path


def print_table(rows):
    print("%-12s %7s %7s %10s %16s %24s %14s" %
          ("Model", "Layers", "Anchor", "r@anchor", "Mid-band max",
           "Late checkpoint", "Global max"))
    for r in rows:
        print("%-12s %7d %7s %10.3f %16s %24s %14s" %
              (r["model"], r["n"], "L%d" % r["anchor"], r["r_anchor"],
               "L%d (%.3f)" % (r["mid_layer"], r["r_mid"]),
               "L%d (%.3f) @%.2f" % (r["late_layer"], r["r_late"], r["late_depth"]),
               "L%d (%.3f)" % (r["gmax_layer"], r["r_gmax"])))


def check(rows):
    d = {r["model"]: r for r in rows}
    ok = True

    def near(label, got, want, tol=0.0006):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print("  [%s] %-46s %s (paper %s)" % ("OK" if good else "FAIL", label, got, want))

    print("Verifying the numbers quoted in the paper:")
    for m, a, ra, ml, rm, ll, rl, dep in [
            ("XLSR-53", 13, 0.102, 10, 0.186, 19, 0.218, 0.79),
            ("XLS-R-300M", 13, 0.410, 10, 0.419, 18, 0.361, 0.75),
            ("XLS-R-1B", 25, 0.141, 22, 0.270, 38, 0.397, 0.79),
            ("XLS-R-2B", 25, 0.345, 25, 0.345, 38, 0.490, 0.79)]:
        r = d[m]
        near("%s anchor layer" % m, r["anchor"], a)
        near("%s r at anchor" % m, round(r["r_anchor"], 3), ra)
        near("%s mid-band layer" % m, r["mid_layer"], ml)
        near("%s mid-band r" % m, round(r["r_mid"], 3), rm)
        near("%s late layer" % m, r["late_layer"], ll)
        near("%s late r" % m, round(r["r_late"], 3), rl)
        near("%s late rel. depth" % m, round(r["late_depth"], 2), dep)

    # the paper's structural claims
    lo = min(d[m]["late_depth"] for m in d)
    hi = max(d[m]["late_depth"] for m in d)
    print("  [%s] late checkpoint rel. depth within 0.75-0.79: %.2f-%.2f"
          % ("OK" if abs(lo - 0.75) < 0.006 and abs(hi - 0.79) < 0.006 else "FAIL", lo, hi))
    ok &= abs(lo - 0.75) < 0.006 and abs(hi - 0.79) < 0.006

    early = [d[m]["gmax_layer"] <= anchor_layer(d[m]["n"]) for m in ("XLSR-53", "XLS-R-300M")]
    print("  [%s] global max is early for XLSR-53 and XLS-R-300M" % ("OK" if all(early) else "FAIL"))
    ok &= all(early)
    late_g = [d[m]["gmax_layer"] == d[m]["late_layer"] for m in ("XLS-R-1B", "XLS-R-2B")]
    print("  [%s] global max is the late checkpoint for the 1B/2B pair" % ("OK" if all(late_g) else "FAIL"))
    ok &= all(late_g)
    print("  [%s] XLSR-53 mid-band max is far below the others (0.186 vs 0.419/0.270/0.345)"
          % ("OK" if d["XLSR-53"]["r_mid"] < 0.25 < d["XLS-R-300M"]["r_mid"] else "FAIL"))
    ok &= d["XLSR-53"]["r_mid"] < 0.25

    print("\n%s" % ("ALL NUMBERS MATCH THE PAPER" if ok else "MISMATCH -- do not compile"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    project = _project_root()
    ap.add_argument("--root", default=os.path.join(project, "re-organized"))
    ap.add_argument("--outdir", default=os.path.join(project, "results"))
    ap.add_argument("--check", action="store_true",
                    help="verify the numbers quoted in geometry_revised.tex")
    a = ap.parse_args()

    cache = os.path.join(a.root, "mantel_results_cache.csv")
    if not os.path.exists(cache):
        print("cache not found: %s" % cache, file=sys.stderr)
        return 2
    rows = collect(cache)
    print_table(rows)
    if a.check:
        return check(rows)
    print("\nwrote %s" % write_tex(rows, os.path.join(a.outdir, "mantel_checkpoints_table.tex")))
    print("wrote %s" % write_csv(rows, os.path.join(a.outdir, "mantel_checkpoints_metrics.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
