#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify the p-values quoted in Section IV-G against the run they came from.

Section IV-G quotes numbers from three separate statistical runs, and the three
are easy to mix up because they differ only in which layer the 300M model uses:

    anchor    --layers 13,13,25,25      -> anchor_13_13_25_25.log
    checkpoint--late                    -> late_checkpoint.log     (Table X layers)
    control   --layers 19,19,38,38      -> control_19_19_38_38.log (300M at L19)

This was written after the checkpoint omnibus values were quoted from the control
run: both are "the checkpoint" in prose, but they are different 300M layers
(L18 vs L19) and different numbers. The checker pins every quoted value to one
specific log.

Usage
-----
    python 43_09_check_section_IV_G_numbers.py
    python 43_09_check_section_IV_G_numbers.py --paper ../paper_revise/geometry_revised.tex

Exit code 0 = every quoted value is present in the log it is attributed to.
"""

import argparse
import os

def _project_root():
    """The directory holding re-organized/. Current directory, then the script's, then its parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)

import re
import sys

ANCHOR = "anchor_13_13_25_25.log"
CHECKPOINT = "late_checkpoint.log"
CONTROL = "control_19_19_38_38.log"

# (log, pair, metric, expected p as printed in the paper)
QUOTES = [
    (ANCHOR, "1B-2B", "Max_Corr", "0.08"),
    (ANCHOR, "1B-2B", "Best_Lag", "0.0029"),
    (ANCHOR, "1B-2B", "ID", "0.098"),
    (ANCHOR, "omnibus", "Best_Lag", "0.034"),
    (ANCHOR, "omnibus", "ID", "3.67e-11"),
    (CHECKPOINT, "1B-2B", "Max_Corr", "0.006"),
    (CHECKPOINT, "1B-2B", "Best_Lag", "1.58e-06"),
    (CHECKPOINT, "1B-2B", "ID", "8.42e-05"),
    (CHECKPOINT, "omnibus", "Best_Lag", "2.23e-10"),
    (CHECKPOINT, "omnibus", "ID", "2.40e-21"),
]

NUM = r"(\d+\.?\d*(?:e[-+]\d+)?)"


def parse(path):
    """{(pair, metric): p} scraped from one run log."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    pair = omnibus = None
    for line in txt.splitlines():
        m = re.search(r"成对 (\S+) vs (\S+)", line)
        if m:
            pair = "1B-2B" if "1B" in m.group(1) + m.group(2) and "2B" in m.group(2) else "other"
            omnibus = None
            continue
        if "综合四模型" in line:
            pair, omnibus = None, "omnibus"
            continue
        if "缩放律" in line:
            pair = omnibus = None
            continue
        if pair and "MW U=" in line or (pair and "Welch" in line):
            metric = ("Max_Corr" if "Max_Corr" in line else
                      "Best_Lag" if "Best_Lag" in line else
                      "ID" if "Intrinsic_Dim" in line else None)
            p = re.search(r"p=" + NUM, line)
            if metric and p:
                out[(pair, metric)] = p.group(1)
        if omnibus and ("Kruskal-Wallis" in line or "ANOVA" in line):
            p = re.search(r"p=" + NUM, line)
            if p:
                # the metric name was printed on the previous line
                out[(omnibus, "last_metric")] = p.group(1)
        if omnibus and re.match(r"\s+(Max_Corr|Best_Lag_Sec|Intrinsic_Dim)\s", line):
            metric = ("Max_Corr" if "Max_Corr" in line else
                      "Best_Lag" if "Best_Lag" in line else "ID")
            out[(omnibus, "pending")] = metric
    # resolve omnibus p-values to their metric using the pending marker order
    return out, txt


def omnibus_map(txt):
    """{metric: p} for the omnibus block, using the line order."""
    out = {}
    metric = None
    for line in txt.splitlines():
        m = re.match(r"\s+(Max_Corr|Best_Lag_Sec|Intrinsic_Dim)\s", line)
        if m and "=" in line and "p=" not in line:
            metric = {"Max_Corr": "Max_Corr", "Best_Lag_Sec": "Best_Lag",
                      "Intrinsic_Dim": "ID"}[m.group(1)]
            continue
        if metric and ("Kruskal-Wallis" in line or "ANOVA" in line):
            p = re.search(r"p=" + NUM, line)
            if p:
                out[metric] = p.group(1)
                metric = None
    return out


def close(a, b, rel=0.02):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if b == 0:
        return abs(a) < 1e-12
    return abs(a - b) / abs(b) <= rel


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    _root = _project_root()
    _logdir = os.path.join(_root, "logs")
    ap.add_argument("--project", default=_logdir if os.path.isdir(_logdir) else _root)
    ap.add_argument("--paper", default=None)
    a = ap.parse_args()

    logs = {}
    for name in (ANCHOR, CHECKPOINT, CONTROL):
        path = os.path.join(a.project, name)
        if not os.path.exists(path):
            print("[ERROR] missing log: %s" % path)
            return 2
        logs[name] = omnibus_map(open(path, encoding="utf-8", errors="replace").read())

    ok = True
    print("Section IV-G p-values, pinned to the run that produced them:\n")
    for log, pair, metric, expected in QUOTES:
        if pair == "omnibus":
            got = logs[log].get(metric)
        else:
            txt = open(os.path.join(a.project, log), encoding="utf-8", errors="replace").read()
            got = None
            block = txt.split("成对 XLS-R-1B vs XLS-R-2B")[-1] if "成对 XLS-R-1B vs XLS-R-2B" in txt else ""
            for line in block.splitlines()[:8]:
                name = ("Max_Corr" if "Max_Corr" in line else
                        "Best_Lag" if "Best_Lag" in line else
                        "ID" if "Intrinsic_Dim" in line else None)
                if name == metric:
                    p = re.search(r"p=" + NUM, line)
                    if p:
                        got = p.group(1)
                    break
        good = close(got, expected)
        ok &= good
        print("  [%s] %-11s %-4s %-9s log=%-9s paper=%s"
              % ("OK" if good else "FAIL", log.replace(".log", ""), pair, metric, got, expected))

    # the two runs must actually disagree, otherwise the pinning is untested
    if close(logs[CONTROL].get("Best_Lag"), logs[CHECKPOINT].get("Best_Lag")):
        ok = False
        print("  [FAIL] control and checkpoint omnibus lag agree -- the pinning is untested")
    else:
        print("  [OK]   control and checkpoint differ on omnibus lag (%s vs %s), so the "
              "attribution matters" % (logs[CONTROL].get("Best_Lag"),
                                       logs[CHECKPOINT].get("Best_Lag")))

    print("\n%s" % ("ALL QUOTED VALUES COME FROM THE STATED RUN" if ok else "ATTRIBUTION ERROR"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
