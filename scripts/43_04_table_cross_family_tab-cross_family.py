#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce Table `tab:cross_family` of geometry_revised.tex.

Seven encoders are compared under one protocol: intrinsic dimension (ID),
normalised manifold area and temporal entropy are read at three matched
*relative* depths (0.35 / 0.50 / 0.65) and averaged into a "band" mean.

Why relative depth
------------------
The encoders do not share a depth (13 / 25 / 49 layers), so absolute layer
indices are not comparable. A relative depth d is mapped to
    layer = int(round(d * (n_layers - 1)))
with n_layers counted *including* the input/embedding layer, which is the same
convention as the twelve-round protocol in 02_07_formant_dim_statistical_test_*.

Why 0.35-0.65
-------------
Both ends of the depth axis are dominated by artefacts: the first layers carry
near-raw acoustic features, and the last two layers of every encoder contract
sharply (see Sec. V-E). The band between 0.35 and 0.65 is where the encoders are
actually comparable.

Inputs (all relative to --root, default ./re-organized):
    <model>/ID_result/ID_Analysis_L<L>_Fixed.csv     ID per language per layer
    manifold_area.csv                                Model,Layer,
                                                     "Physical Coverage Area",
                                                     "Normalized Manifold Area"
    Entropy_Results_<tag>_moreLayers.csv             Layer_Int,Entropy

Outputs:
    tables/cross_family_table.tex                    the LaTeX table body+header
    tables/cross_family_metrics.csv                  every number, one row/model
    stdout                                           the same table as plain text

The script uses only the standard library so that it runs in any environment
that can read the CSVs. It does not import pandas/scipy.

Usage
-----
    python 43_04_table_cross_family_tab-cross_family.py           # write + print
    python 43_04_table_cross_family_tab-cross_family.py --check   # verify the paper
    python 43_04_table_cross_family_tab-cross_family.py --root PATH

Defaults are resolved from the script location (.../model_compare01), so it can
be run from any working directory. The generated .tex is what the paper inputs
via \\input{tables/cross_family_table.tex}.

Numbers quoted in geometry_revised.tex (verified by --check):
    ID band means      11.72 / 14.73 / 14.97 / 14.99 / 16.59 / 16.81 / 17.04
    300M tier range    14.73-14.99 across 53, 128 and 1100+ training languages
    1B tier range      16.59-17.04
    step within XLS-R  +1.62   (14.97 -> 16.59)
    step within MMS    +1.82   (14.99 -> 16.81)
    area ratio 300M    MMS/XLS-R = 0.0846/0.0332 = 2.55x
    area ratio 1B      MMS/XLS-R = 0.0495/0.0315 = 1.57x
    entropy range      1.58-2.16, no ordering
    mHuBERT-147 depth  12.96 -> 11.83 -> 10.38 (monotone decrease)
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

# display name, directory, manifold_area.csv key, entropy file tag, params (M),
# training languages. mHuBERT-147: utterance/facebook HuBERT-147, 147 languages.
# MMS: 1100+ languages (MMS paper reports >1100), base self-supervised weights.
MODELS = [
    ("XLSR-53",     "wav2vec2-large-xlsr-53", "XLSR-53",    "XLSR_53",     300,  53),
    ("XLS-R-300M",  "wav2vec2-xls-r-300m",    "XLS-R-300m", "XLS_R_300M",  300, 128),
    ("XLS-R-1B",    "wav2vec2-xls-r-1b",      "XLS-R-1B",   "XLS_R_1B",   1000, 128),
    ("XLS-R-2B",    "wav2vec2-xls-r-2b",      "XLS-R-2B",   "XLS_R_2B",   2000, 128),
    ("MMS-300M",    "mms-300m",               "MMS-300M",   "MMS_300M",    300, 1100),
    ("MMS-1B",      "mms-1b",                 "MMS-1B",     "MMS_1B",     1000, 1100),
    ("mHuBERT-147", "mHuBERT-147",            "mHuBERT-147", "mHuBERT_147", 95, 147),
]

REL_DEPTHS = (0.35, 0.50, 0.65)


def count_layers(model_dir):
    """Number of layers including the input layer (0..n-1)."""
    d = os.path.join(model_dir, "ID_result")
    return max(int(f.split("_L")[1].split("_")[0])
               for f in os.listdir(d) if f.startswith("ID_Analysis_L")) + 1


def rel_layer(rel, n_layers):
    return int(round(rel * (n_layers - 1)))


def read_id(model_dir, layer):
    """Mean intrinsic dimension over the 24 FLEURS languages at one layer."""
    p = os.path.join(model_dir, "ID_result", "ID_Analysis_L%d_Fixed.csv" % layer)
    if not os.path.exists(p):
        return None, 0
    vals = [float(r["Intrinsic_Dim"]) for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["Language"].endswith("_fleurs")]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def read_area(path):
    """{model_lower: {layer: normalised area}}"""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(r["Model"].strip().lower(), {})[int(r["Layer"])] = \
            float(r["Normalized Manifold Area"])
    return out


def read_entropy(root, tag):
    p = os.path.join(root, "Entropy_Results_%s_moreLayers.csv" % tag)
    if not os.path.exists(p):
        return {}
    return {int(r["Layer_Int"]): float(r["Entropy"])
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def collect(root):
    area = read_area(os.path.join(root, "manifold_area.csv"))
    rows = []
    for disp, mdir, area_key, tag, params, langs in MODELS:
        mdir_full = os.path.join(root, mdir)
        n = count_layers(mdir_full)
        layers = [rel_layer(d, n) for d in REL_DEPTHS]
        ids = [read_id(mdir_full, L)[0] for L in layers]
        ent = read_entropy(root, tag)
        hs = [ent.get(L) for L in layers]
        a = area.get(area_key.lower(), {})
        ars = [a.get(L) for L in layers]
        rows.append(dict(model=disp, params=params, langs=langs, n_layers=n,
                         layers=layers, id=ids, id_band=mean(ids),
                         h=hs, h_band=mean(hs), area=ars, area_band=mean(ars)))
    return rows


def fmt_langs(k):
    return "1100+" if k > 1000 else str(k)


def fmt_params(p):
    return "%dM" % p if p < 1000 else "%dB" % (p // 1000)


def write_tex(rows, path):
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{The same relative-depth protocol applied to seven encoders. "
        r"Intrinsic dimension $\widehat{id}$ is the mean over the 24 FLEURS languages at "
        r"relative depth $0.35/0.50/0.65$; the band mean averages the three. "
        r"$\widehat{A}_{\text{norm}}$ is the normalised hull area averaged over the same "
        r"three layers. Layers are matched by relative depth, not by index, because the "
        r"encoders have 13, 25 or 49 layers. MMS encoders are the base self-supervised "
        r"weights (not the adaptor fine-tunes).}"
        r"\label{tab:cross_family}",
        r"\small",
        r"\begin{tabular}{l r r c c c c r}",
        r"\toprule",
        r"\textbf{Encoder} & \textbf{Params} & \textbf{Train langs} & \multicolumn{3}{c}{$\widehat{id}$ at rel.\ depth} "
        r"& \textbf{$\widehat{id}$ band} & \textbf{$\widehat{A}_{\text{norm}}$ band} \\",
        r"\cmidrule(lr){4-6}",
        r" & & & 0.35 & 0.50 & 0.65 & & \\",
        r"\midrule",
    ]
    prev_group = None
    for r in rows:
        group = "xlsr" if r["model"].startswith(("XLSR", "XLS-R")) else "other"
        if prev_group == "xlsr" and group == "other":
            lines.append(r"\midrule")
        prev_group = group
        cells = [r["model"], fmt_params(r["params"]), fmt_langs(r["langs"])]
        cells += ["%.2f" % v for v in r["id"]]
        cells += ["%.2f" % r["id_band"], "%.4f" % r["area_band"]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Encoder", "Params_M", "Train_langs", "N_layers",
                    "Layer_0.35", "Layer_0.50", "Layer_0.65",
                    "ID_0.35", "ID_0.50", "ID_0.65", "ID_band_mean",
                    "Hn_band_mean", "NormArea_band_mean"])
        for r in rows:
            w.writerow([r["model"], r["params"], r["langs"], r["n_layers"]]
                       + r["layers"] + ["%.4f" % v for v in r["id"]]
                       + ["%.4f" % r["id_band"], "%.4f" % r["h_band"],
                          "%.4f" % r["area_band"]])
    return path


def print_table(rows):
    print("%-12s %6s %6s %8s %8s %8s %8s %9s" %
          ("Encoder", "Params", "Langs", "ID .35", "ID .50", "ID .65", "ID band", "Anorm"))
    for r in rows:
        print("%-12s %6s %6s %8.2f %8.2f %8.2f %8.2f %9.4f" %
              (r["model"], fmt_params(r["params"]), fmt_langs(r["langs"]),
               r["id"][0], r["id"][1], r["id"][2], r["id_band"], r["area_band"]))


def check(rows):
    """Assert every number quoted in geometry_revised.tex."""
    d = {r["model"]: r for r in rows}
    ok = True

    def near(label, got, want, tol=0.006):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print("  [%s] %-42s %.4f (paper %.4f)" % ("OK" if good else "FAIL", label, got, want))

    print("Verifying the numbers quoted in the paper:")
    for m, want in [("mHuBERT-147", 11.72), ("XLSR-53", 14.73), ("XLS-R-300M", 14.97),
                    ("MMS-300M", 14.99), ("XLS-R-1B", 16.59), ("MMS-1B", 16.81),
                    ("XLS-R-2B", 17.04)]:
        near("ID band mean, %s" % m, d[m]["id_band"], want)

    near("step within XLS-R 300M->1B", d["XLS-R-1B"]["id_band"] - d["XLS-R-300M"]["id_band"], 1.62, 0.02)
    near("step within MMS 300M->1B", d["MMS-1B"]["id_band"] - d["MMS-300M"]["id_band"], 1.82, 0.02)
    near("area ratio MMS/XLS-R at 300M",
         d["MMS-300M"]["area_band"] / d["XLS-R-300M"]["area_band"], 2.55, 0.03)
    near("area ratio MMS/XLS-R at 1B",
         d["MMS-1B"]["area_band"] / d["XLS-R-1B"]["area_band"], 1.57, 0.03)

    tier300 = [d[m]["id_band"] for m in ("XLSR-53", "XLS-R-300M", "MMS-300M")]
    tier1b = [d[m]["id_band"] for m in ("XLS-R-1B", "XLS-R-2B", "MMS-1B")]
    near("300M tier spread", max(tier300) - min(tier300), 0.26, 0.01)
    near("1B tier spread", max(tier1b) - min(tier1b), 0.45, 0.01)

    hs = [r["h_band"] for r in rows]
    near("entropy band min", min(hs), 1.58, 0.01)
    near("entropy band max", max(hs), 2.16, 0.01)
    print("  [%s] mHuBERT-147 ID decreases with depth: %.2f -> %.2f -> %.2f"
          % ("OK" if d["mHuBERT-147"]["id"][0] > d["mHuBERT-147"]["id"][1] >
             d["mHuBERT-147"]["id"][2] else "FAIL",
             *d["mHuBERT-147"]["id"]))
    ok &= (d["mHuBERT-147"]["id"][0] > d["mHuBERT-147"]["id"][1] >
           d["mHuBERT-147"]["id"][2])
    print("\n%s" % ("ALL NUMBERS MATCH THE PAPER" if ok else "MISMATCH -- do not compile"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    project = _project_root()                      # .../model_compare01
    ap.add_argument("--root", default=os.path.join(project, "re-organized"),
                    help="data root holding the per-model result directories")
    ap.add_argument("--outdir", default=os.path.join(project, "results"),
                    help="where to write the table/CSV (default: paper_revise/tables)")
    ap.add_argument("--check", action="store_true",
                    help="verify the numbers quoted in geometry_revised.tex")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print("data root not found: %s" % a.root, file=sys.stderr)
        return 2
    rows = collect(a.root)
    print_table(rows)
    if a.check:
        return check(rows)
    print("\nwrote %s" % write_tex(rows, os.path.join(a.outdir, "cross_family_table.tex")))
    print("wrote %s" % write_csv(rows, os.path.join(a.outdir, "cross_family_metrics.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
