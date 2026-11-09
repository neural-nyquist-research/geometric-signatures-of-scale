#!/usr/bin/env python3
"""Check the topology numbers quoted in Section IV-D against the shipped run.

Reads re-organized/wav2vec2-xls-r-2b/topology_quant_results.csv, the output of
11_01_ripser_persim.py (ripser, maxdim=1; the combined score is the Euclidean
combination of the H0 and H1 Wasserstein distances for one language pair), and
verifies the three layer means and the paired change that the manuscript quotes:

    L32 = 20.62   L40 = 16.90   L46 = 25.56   paired L40-L32 = -3.73, 95% CI [-4.84, -2.66]

The interval is a percentile bootstrap over the 300 language pairs, resampled
independently at the two layers; the seed is fixed so the check is deterministic.

    python 43_13_check_topology_numbers.py            # verify
    python 43_13_check_topology_numbers.py --verbose  # print the full profile

Exit status: 0 when every quoted value matches, 1 when one drifts, 2 when the
CSV is missing.
"""
import argparse
import csv
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

QUOTED = {
    "mean_L32": 20.62,
    "mean_L40": 16.90,
    "mean_L46": 25.56,
    "paired_L40_minus_L32": -3.73,
    "ci_low": -4.84,
    "ci_high": -2.66,
}
TOL = 0.01


def _project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)


def load(path):
    per_layer = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            per_layer[int(row["layer"])][(row["lang_a"], row["lang_b"])] = float(
                row["total_topo_score"])
    return per_layer


def bootstrap_ci(diffs, n_boot=4000, seed=0):
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser(description="Verify the Section IV-D topology numbers.")
    ap.add_argument("--csv", default=os.path.join(
        _project_root(), "re-organized", "wav2vec2-xls-r-2b", "topology_quant_results.csv"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] missing {args.csv}")
        print("        run 11_01_ripser_persim.py on the XLS-R-2B activations, or restore the "
              "shipped copy.")
        return 2

    per = load(args.csv)
    layers = sorted(per)
    means = {L: statistics.mean(per[L].values()) for L in layers}

    if args.verbose:
        print("layer  mean_total_topo_score  pairs")
        for L in layers:
            print(f"{L:5d}  {means[L]:20.2f}  {len(per[L]):5d}")

    pairs = sorted(per[32])
    diffs = [per[40][k] - per[32][k] for k in pairs]
    lo, hi = bootstrap_ci(diffs)

    got = {
        "mean_L32": means[32],
        "mean_L40": means[40],
        "mean_L46": means[46],
        "paired_L40_minus_L32": statistics.mean(diffs),
        "ci_low": lo,
        "ci_high": hi,
    }

    bad = []
    print(f"layers {layers[0]}..{layers[-1]}, {len(pairs)} language pairs per layer")
    for key, want in QUOTED.items():
        have = got[key]
        ok = abs(have - want) <= TOL
        print(f"[{'OK' if ok else 'FAIL'}] {key:24s} paper {want:+.2f}  data {have:+.2f}")
        if not ok:
            bad.append(key)

    if bad:
        print("\nDRIFT: the manuscript and the data disagree on " + ", ".join(bad))
        return 1
    print("\nALL QUOTED TOPOLOGY VALUES MATCH THE SHIPPED RUN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
