#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Corpus-robustness and data-integrity check: FLEURS (24 languages) vs MLS (8).

The paper reports FLEURS throughout and uses MLS only as a cross-corpus control.
This script produces the numbers that justify that decision, and it refuses to
compare any model whose ID output fails an integrity test.

Three things are computed.

1. Integrity of the ID extraction, per model per corpus. Two failure modes are
   caught, both of which have occurred in real runs on this project:
     * over-run  -- more layer files than the encoder has layers (a 25-layer
                    model with 49 files means the layer loop used another
                    model's configuration);
     * duplicate -- layer files byte-identical to another model's, i.e. one
                    model's results copied into another model's directory.
   A model failing either test is reported as EXCLUDED and never enters a
   comparison. The duplicate test hashes the files rather than comparing counts,
   because a copy is indistinguishable from a real run by size alone.

2. Node scarcity. A node is a dimension surviving |Max_Corr| > 0.1, the filter
   the geometry uses. Convex-hull area over three nodes is a segment, and
   histogram entropy is upward-biased when nodes are few, so hull and entropy
   cannot be pooled across corpora with 8 and 24 languages.

3. Intrinsic dimension across corpora, computed only where both roots pass the
   integrity test. ID is estimated per language and does not depend on how many
   dimensions survive, so it is the only cross-corpus quantity.

Usage
-----
    python 43_06_corpus_check_fleurs_vs_mls.py
    python 43_06_corpus_check_fleurs_vs_mls.py --check

Standard library only.
"""

import argparse
import collections
import csv
import hashlib
import os

def _project_root():
    """The directory holding re-organized/. Current directory, then the script's, then its parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)

import sys

# display name, directory, expected layers INCLUDING the input layer
MODELS = [("XLSR-53", "wav2vec2-large-xlsr-53", 25),
          ("XLS-R-300M", "wav2vec2-xls-r-300m", 25),
          ("XLS-R-1B", "wav2vec2-xls-r-1b", 49),
          ("XLS-R-2B", "wav2vec2-xls-r-2b", 49),
          ("MMS-300M", "mms-300m", 25),
          ("MMS-1B", "mms-1b", 49),
          ("mHuBERT-147", "mHuBERT-147", 13)]

NODE_THRESHOLD = 0.1
PROTOCOL_DEPTHS = (0.35, 0.50, 0.65)   # the three layers of Table IV (cross-family)
BAND_HALF_WIDTH = 0.16          # relative depth 0.34-0.66 around the midpoint


# ---------------------------------------------------------------- integrity ---

def id_files(root, model_dir):
    d = os.path.join(root, model_dir, "ID_result")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith("ID_Analysis_L"))


def fingerprint(root, model_dir):
    """Digest of the ID directory: first file, last file, and the count."""
    fs = id_files(root, model_dir)
    if not fs:
        return None
    d = os.path.join(root, model_dir, "ID_result")
    h = hashlib.md5()
    for f in fs:
        h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def integrity(root):
    """{model_dir: (status, detail)}; status in ok / partial / over-run / duplicate / absent."""
    out = {}
    prints = {md: fingerprint(root, md) for _d, md, _n in MODELS}
    for _disp, md, expected in MODELS:
        fs = id_files(root, md)
        if not fs:
            out[md] = ("absent", "no ID_result")
            continue
        n = len(fs)
        dup = [m for m, p in prints.items() if m != md and p is not None and p == prints[md]]
        if dup:
            # a copy is the directory whose file count contradicts its own depth;
            # if both counts are consistent, neither can be singled out
            own_ok = (n == expected)
            others_ok = all(len(id_files(root, m)) == dict((x[1], x[2]) for x in MODELS)[m]
                            for m in dup)
            if own_ok and not others_ok:
                out[md] = ("ok", "%d/%d layers (identical to %s, which over-runs its depth)"
                           % (n, expected, ", ".join(sorted(dup))))
                continue
            out[md] = ("duplicate", "byte-identical to %s" % ", ".join(sorted(dup)))
        elif n > expected:
            out[md] = ("over-run", "%d files for a %d-layer encoder" % (n, expected))
        elif n < expected:
            out[md] = ("partial", "%d/%d layers" % (n, expected))
        else:
            out[md] = ("ok", "%d/%d layers" % (n, expected))
    return out


# ------------------------------------------------------------------ metrics ---

def nodes_per_layer(root, model_dir):
    """{layer: number of dimensions with |Max_Corr| > 0.1}"""
    p = os.path.join(root, model_dir, "formant_dim_analysis_cache",
                     "Formant_Lag_Analysis_Results_moreLayers.csv")
    if not os.path.exists(p):
        return {}
    c = collections.Counter()
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if abs(float(r["Max_Corr"])) > NODE_THRESHOLD:
            c[int(r["Layer"])] += 1
    n = max(c) + 1 if c else 0
    return {L: c.get(L, 0) for L in range(n)}


def id_band(root, model_dir, suffix, n_layers):
    """Band mean of ID over relative depth 0.34-0.66, or None if unusable."""
    d = os.path.join(root, model_dir, "ID_result")
    band = [L for L in range(n_layers)
            if abs(L / (n_layers - 1) - 0.5) <= BAND_HALF_WIDTH]
    vals = []
    for L in band:
        p = os.path.join(d, "ID_Analysis_L%d_Fixed.csv" % L)
        if not os.path.exists(p):
            return None
        v = [float(r["Intrinsic_Dim"])
             for r in csv.DictReader(open(p, encoding="utf-8"))
             if r["Language"].endswith(suffix)]
        if v:
            vals.append(sum(v) / len(v))
    return sum(vals) / len(vals) if vals else None


def id_values(root, model_dir, suffix, n_layers, depths=PROTOCOL_DEPTHS):
    """All per-language ID values at the three relative depths of Table IV."""
    d = os.path.join(root, model_dir, "ID_result")
    out = []
    for rel in depths:
        L = int(round(rel * (n_layers - 1)))
        p = os.path.join(d, "ID_Analysis_L%d_Fixed.csv" % L)
        if not os.path.exists(p):
            return None
        out += [float(r["Intrinsic_Dim"])
                for r in csv.DictReader(open(p, encoding="utf-8"))
                if r["Language"].endswith(suffix)]
    return out


def stats(v):
    """Mean and standard error of a list."""
    n = len(v)
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    return m, (var / n) ** 0.5


def gap(v1, v2):
    """mean(v2) - mean(v1) with a 95% CI (unpaired, normal approximation)."""
    m1, s1 = stats(v1)
    m2, s2 = stats(v2)
    d = m2 - m1
    se = (s1 ** 2 + s2 ** 2) ** 0.5
    return d, d - 1.96 * se, d + 1.96 * se


def tier_stats(project):
    """{corpus: {model: values}} at the Table IV depths, for the four tier models."""
    roots = {"FLEURS": (os.path.join(project, "re-organized"), "_fleurs"),
             "MLS": (os.path.join(project, "re-organized_mls"), "_mls")}
    keep = ["XLS-R-300M", "XLS-R-1B", "MMS-300M", "MMS-1B"]
    out = {}
    for corpus, (root, suffix) in roots.items():
        out[corpus] = {}
        for disp, md, n in MODELS:
            if disp not in keep:
                continue
            v = id_values(root, md, suffix, n)
            if v:
                out[corpus][disp] = v
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def collect(project):
    roots = {"FLEURS": os.path.join(project, "re-organized"),
             "MLS": os.path.join(project, "re-organized_mls")}
    integ = {k: integrity(v) for k, v in roots.items()}
    rows = []
    for disp, md, n in MODELS:
        row: dict = dict(model=disp, model_dir=md, n=n)
        for corpus, suffix in (("FLEURS", "_fleurs"), ("MLS", "_mls")):
            status, detail = integ[corpus][md]
            row["status_" + corpus] = status
            row["detail_" + corpus] = detail
            nd = nodes_per_layer(roots[corpus], md)
            row["nodes_" + corpus] = nd
            row["id_" + corpus] = (id_band(roots[corpus], md, suffix, n)
                                   if status == "ok" else None)
        rows.append(row)
    return rows


def print_tiers(project):
    t = tier_stats(project)
    print("\n=== 4. Tier structure at the Table IV depths (.35/.50/.65) ===")
    for corpus in ("FLEURS", "MLS"):
        if not t.get(corpus):
            continue
        print("  %s" % corpus)
        for m in ("XLS-R-300M", "MMS-300M", "MMS-1B", "XLS-R-1B"):
            if m in t[corpus]:
                mu, se = stats(t[corpus][m])
                print("    %-12s %.2f (SE %.2f, n=%d)" % (m, mu, se, len(t[corpus][m])))
        for a, b in (("XLS-R-300M", "XLS-R-1B"), ("XLS-R-300M", "MMS-1B"),
                     ("XLS-R-300M", "MMS-300M")):
            if a in t[corpus] and b in t[corpus]:
                d, lo, hi = gap(t[corpus][a], t[corpus][b])
                star = "*" if (lo > 0 or hi < 0) else " "
                print("    %-24s %+6.2f  [%+.2f, %+.2f] %s" % ("%s-%s" % (b, a), d, lo, hi, star))


def print_tables(rows):
    print("=== 1. ID extraction integrity ===")
    print("%-12s %-34s %-34s" % ("Model", "FLEURS", "MLS"))
    for r in rows:
        print("%-12s %-34s %-34s" % (r["model"],
                                     "%s (%s)" % (r["status_FLEURS"], r["detail_FLEURS"]),
                                     "%s (%s)" % (r["status_MLS"], r["detail_MLS"])))

    print("\n=== 2. Node scarcity (|Max_Corr|>0.1) ===")
    print("%-12s %20s %20s %8s %14s" %
          ("Model", "FLEURS nodes/layer", "MLS nodes/layer", "ratio", "MLS <=5 nodes"))
    for r in rows:
        f, m = r["nodes_FLEURS"], r["nodes_MLS"]
        if not f or not m:
            print("%-12s %20s %20s" % (r["model"], "-", "-"))
            continue
        fm, mm = sum(f.values()) / len(f), sum(m.values()) / len(m)
        print("%-12s %20s %20s %8.1f %14s" %
              (r["model"], "%.1f (med %d)" % (fm, median(list(f.values()))),
               "%.1f (med %d)" % (mm, median(list(m.values()))), fm / mm,
               "%d/%d" % (sum(1 for v in m.values() if v <= 5), len(m))))

    print("\n=== 3. Intrinsic dimension across corpora (integrity-passing models only) ===")
    print("%-12s %12s %10s %8s" % ("Model", "FLEURS", "MLS", "shift"))
    for r in rows:
        a, b = r["id_FLEURS"], r["id_MLS"]
        if a is not None and b is not None:
            print("%-12s %12.2f %10.2f %+8.2f" % (r["model"], a, b, b - a))
        else:
            why = []
            if a is None:
                why.append("FLEURS %s" % r["status_FLEURS"])
            if b is None:
                why.append("MLS %s (%s)" % (r["status_MLS"], r["detail_MLS"]))
            print("%-12s   excluded: %s" % (r["model"], "; ".join(why)))


def check(rows):
    d = {r["model"]: r for r in rows}
    ok = True

    def near(label, got, want, tol):
        nonlocal ok
        good = got is not None and abs(got - want) <= tol
        ok &= good
        print("  [%s] %-52s %s (paper %s)" %
              ("OK" if good else "FAIL", label,
               "n/a" if got is None else "%.2f" % got, want))

    print("Verifying the numbers quoted in the paper:")
    f_med = [median(list(r["nodes_FLEURS"].values())) for r in rows if r["nodes_FLEURS"]]
    m_med = [median(list(r["nodes_MLS"].values())) for r in rows if r["nodes_MLS"]]
    near("FLEURS median nodes/layer, lowest", min(f_med), 47, 1)
    near("FLEURS median nodes/layer, highest", max(f_med), 84, 1)
    near("MLS median nodes/layer, lowest", min(m_med), 7, 1)
    near("MLS median nodes/layer, highest", max(m_med), 22, 1)
    ratios = []
    for r in rows:
        f, m = r["nodes_FLEURS"], r["nodes_MLS"]
        if f and m:
            ratios.append((sum(f.values()) / len(f)) / (sum(m.values()) / len(m)))
    near("node ratio FLEURS/MLS, lowest", min(ratios), 2.5, 0.05)
    near("node ratio FLEURS/MLS, highest", max(ratios), 6.6, 0.05)
    no_le5 = all(sum(1 for v in r["nodes_FLEURS"].values() if v <= 5) == 0
                 for r in rows if r["nodes_FLEURS"])
    print("  [%s] FLEURS has no layer with <=5 nodes" % ("OK" if no_le5 else "FAIL"))
    ok &= no_le5
    le5 = [r for r in rows if r["nodes_MLS"] and any(v <= 5 for v in r["nodes_MLS"].values())]
    print("  [%s] MLS has layers with <=5 nodes in %d of %d models"
          % ("OK" if len(le5) == len(rows) else "FAIL", len(le5), len(rows)))
    ok &= len(le5) == len(rows)

    # tier structure at the Table IV depths (the numbers quoted in Sec. V-I and Limitation 3)
    t = tier_stats(os.path.join(_project_root()))
    for corpus, want in (("FLEURS", {"XLS-R-300M": 14.97, "XLS-R-1B": 16.59,
                                     "MMS-300M": 14.99, "MMS-1B": 16.81}),
                         ("MLS", {"XLS-R-300M": 16.04, "XLS-R-1B": 17.89,
                                  "MMS-300M": 15.27, "MMS-1B": 16.43})):
        for model, val in want.items():
            got = stats(t[corpus][model])[0] if model in t.get(corpus, {}) else None
            near("%s ID, three depths, %s" % (model, corpus), got, val, 0.02)

    for corpus, want in (("FLEURS", {"1B-300M": 1.62, "MMS1B-300M": 1.85, "MMS300M-300M": 0.03}),
                         ("MLS", {"1B-300M": 1.85, "MMS1B-300M": 0.39, "MMS300M-300M": -0.78})):
        c = t[corpus]
        pairs = {"1B-300M": ("XLS-R-300M", "XLS-R-1B"),
                 "MMS1B-300M": ("XLS-R-300M", "MMS-1B"),
                 "MMS300M-300M": ("XLS-R-300M", "MMS-300M")}
        for key, val in want.items():
            a, b = pairs[key]
            got = gap(c[a], c[b])[0]
            near("%s gap %s" % (corpus, key), got, val, 0.02)

    print("\n%s" % ("ALL NUMBERS MATCH THE PAPER" if ok else "MISMATCH -- do not compile"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--project", default=_project_root())
    ap.add_argument("--check", action="store_true",
                    help="verify the numbers quoted in geometry_revised.tex")
    a = ap.parse_args()

    for root in ("re-organized", "re-organized_mls"):
        if not os.path.isdir(os.path.join(a.project, root)):
            print("missing corpus root: %s" % root, file=sys.stderr)
            return 2
    rows = collect(a.project)
    print_tables(rows)
    print_tiers(a.project)
    if a.check:
        return check(rows)
    bad = [(r["model"], c, r["status_" + c]) for r in rows
           for c in ("FLEURS", "MLS") if r["status_" + c] not in ("ok", "absent")]
    if bad:
        print("\n%d model/corpus pair(s) excluded from comparison (see table 1)." % len(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
