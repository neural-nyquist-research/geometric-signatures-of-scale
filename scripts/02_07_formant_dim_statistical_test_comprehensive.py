# -*- coding: utf-8 -*-
"""
02_07_formant_dim_statistical_test_comprehensive.py
===================================================
Cyclic statistical test for "relative depth" across four wav2vec2-based models (XLSR-53 / XLS-R-300M / XLS-R-1B / XLS-R-2B) — self-contained version covering all six metrics:

  1. Max_Corr       Per-node (per-dimension) correlation magnitude    —— Can be tested per node
  2. Best_Lag_Sec   Per-node optimal lag              —— Can be tested per node
  3. Intrinsic_Dim  Per-language intrinsic dimensionality (ID)          —— Can be tested per language
  4. Nodes          Number of significant dimensions per layer (single value)        —— Descriptive only
  5. Area           Convex hull physical area (per layer)          —— bootstrap CI + difference test
  6. NormArea       Normalized manifold area (per layer)        —— Same as above

Data Source (FLEURS root directory re-organized/ only; this script does NOT read MLS data):
  <root>/<model>/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv
         Columns: Language, Layer, Dimension, Formant, Max_Corr, Best_Lag_Sec
         nodes = count of rows in that layer where |Max_Corr| > 0.1
  <root>/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv
         Columns: Language, Intrinsic_Dim ; only *_fleurs languages are included (excluding GLOBAL_MEAN, etc.)
  <root>/manifold_area.csv
         Columns: Model, Layer, Physical Coverage Area, Normalized Manifold Area
         (In CSV, '300M' is written as 'XLS-R-300m'; matching is case-insensitive)

Usage:
    python 02_07_formant_dim_statistical_test_comprehensive.py              # Default 8 rounds
    python 02_07_formant_dim_statistical_test_comprehensive.py --loops 12   # Custom number of rounds
    python 02_07_formant_dim_statistical_test_comprehensive.py --no-log     # Do not write log file

Note: This script is fixed to analyze FLEURS (24 languages) data only. It does not support --root and will not fallback to the MLS directory.

Dependencies: numpy is required; scipy is optional (if missing, pure Python implementations are used, p-values may be n/a).
"""

import os
import sys
import math
import csv
import json
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None


# ---------------------------------------------------------------------------
# Model Metadata
# ---------------------------------------------------------------------------
MODELS = [
    ("XLSR-53",    "wav2vec2-large-xlsr-53"),
    ("XLS-R-300M", "wav2vec2-xls-r-300m"),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b"),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b"),
]
NAMES = [m[0] for m in MODELS]
PARAMS_M = {"XLSR-53": 300, "XLS-R-300M": 300, "XLS-R-1B": 1000, "XLS-R-2B": 2000}

# True scaling comparison (same 128 languages, different parameters); another pair has same params but different language counts
SCALING_PAIR = ("XLS-R-1B", "XLS-R-2B")
LANG_PAIR = ("XLSR-53", "XLS-R-300M")

# This script is for FLEURS (24 languages) data analysis only: data root is fixed to re-organized/
# It does not read or fallback to the MLS directory (re-organized_mls/).
FLEURS_ROOT_NAME = "re-organized"
_ROOT_CANDIDATES = [FLEURS_ROOT_NAME]
DATA_ROOT = os.path.join(HERE, FLEURS_ROOT_NAME)


def pick_root():
    """Returns the FLEURS data root directory <script_dir>/re-organized/.

    Does not perform any MLS fallback: if this directory is missing, it is treated as a configuration error, and the caller should exit with an error.
    """
    return os.path.join(HERE, FLEURS_ROOT_NAME)


def count_languages(root):
    """Returns the number of distinct Languages in the first model's CSV (for dataset identity verification)."""
    path = os.path.join(root, _model_dir(NAMES[0]), "formant_dim_analysis_cache",
                        "Formant_Lag_Analysis_Results_moreLayers.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            langs = {str(r.get("Language", "")).strip() for r in csv.DictReader(f)}
        langs.discard("")
        return len(langs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statistical Helper Functions (gracefully degrade if scipy is missing)
# ---------------------------------------------------------------------------
def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _betainc_reg(a, b, x):
    """Regularized incomplete Beta function (for exact t-distribution p-values)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    def cf(aa, bb, xx):
        tiny, f, C, D = 1e-30, 1.0, 1.0, 0.0
        for i in range(300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = m * (bb - m) * xx / ((aa + 2 * m - 1) * (aa + 2 * m))
            else:
                num = -(aa + m) * (aa + bb + m) * xx / ((aa + 2 * m) * (aa + 2 * m + 1))
            D = 1.0 + num * D
            if abs(D) < tiny:
                D = tiny
            D = 1.0 / D
            C = 1.0 + num / C
            if abs(C) < tiny:
                C = tiny
            f *= C * D
            if abs(1.0 - C * D) < 1e-12:
                break
        return f
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x < (a + 1) / (a + b + 2):
        return front * cf(a, b, x)
    return 1.0 - math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta) / b * cf(b, a, 1 - x)


def pearson_exact(x, y):
    """Pearson r + exact two-sided p (Student t, df=n-2). For n=4, df=2, critical |r|=0.95."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    df = n - 2
    if abs(r) >= 1:
        return r, 0.0
    t = abs(r) * math.sqrt(df / (1 - r * r))
    if df == 2:
        p = 1.0 - t / math.sqrt(2.0 + t * t)
    elif df == 1:
        p = 1.0 - (2 / math.pi) * math.atan(t)
    else:
        p = _betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return r, max(0.0, min(1.0, p))


def shapiro_wilk_pvalue(x):
    x = np.asarray(x, dtype=float)
    if sp_stats is not None and len(x) >= 3:
        return sp_stats.shapiro(x)
    return (None, None)


def mann_whitney_u(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None, None
    if sp_stats is not None:
        return sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = np.concatenate([a, b]).argsort().argsort() + 1.0
    U = n1 * n2 + n1 * (n1 + 1) / 2.0 - r[:n1].sum()
    return U, None


def welch_ttest(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    if sp_stats is not None:
        return sp_stats.ttest_ind(a, b, equal_var=False)
    return None, None


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp <= 0:
        return float("nan")
    return (a.mean() - b.mean()) / math.sqrt(sp)


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def ci_cohen_d(d, n1, n2):
    if d is None or not np.isfinite(d) or n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    se = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))
    return d - 1.96 * se, d + 1.96 * se


def ci_cliffs(d, n1, n2):
    if d is None or not np.isfinite(d) or n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    se = math.sqrt((n1 + n2) / (n1 * n2))
    return d - 1.96 * se, d + 1.96 * se


def kruskal_wallis_pvalue(groups):
    data = [np.asarray(g, float) for g in groups if len(g) > 0]
    if len(data) < 2:
        return None, None
    if sp_stats is not None:
        return sp_stats.kruskal(*data)
    return None, None


def anova_ftest(groups):
    data = [np.asarray(g, float) for g in groups if len(g) > 0]
    if len(data) < 2:
        return None, None
    if sp_stats is not None:
        return sp_stats.f_oneway(*data)
    return None, None


# ---------------------------------------------------------------------------
# Convex Hull Area / Bootstrap
# ---------------------------------------------------------------------------
def convex_hull(points):
    """Andrew monotone chain. points: (N,2) -> hull vertices."""
    P = sorted(map(tuple, points))
    if len(P) < 3:
        return P
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_area(points):
    if len(points) < 3:
        return 0.0
    if ConvexHull is not None:
        try:
            return float(ConvexHull(np.asarray(points, float)).volume)
        except Exception:
            pass
    h = convex_hull(points)
    if len(h) < 3:
        return 0.0
    s = 0.0
    for i in range(len(h)):
        x1, y1 = h[i]
        x2, y2 = h[(i + 1) % len(h)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def bootstrap_area_ci(points, n_boot=2000, seed=0):
    """Point estimate = convex hull of full point cloud; CI = bootstrap percentiles."""
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    obs = hull_area(pts)
    rng = np.random.default_rng(seed)
    vals = np.array([hull_area(pts[rng.integers(0, len(pts), len(pts))])
                     for _ in range(n_boot)])
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_area_diff(a, b, n_boot=2000, seed=0):
    """Bootstrap of difference between two models' convex hull areas: (obs_diff, lo, hi)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan"), float("nan")
    obs = hull_area(a) - hull_area(b)
    rng = np.random.default_rng(seed)
    vals = np.array([hull_area(a[rng.integers(0, len(a), len(a))])
                     - hull_area(b[rng.integers(0, len(b), len(b))])
                     for _ in range(n_boot)])
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def _model_dir(name):
    return next(d for n, d in MODELS if n == name)


def load_point_cloud(name, layer):
    """Max_Corr / Best_Lag per node (|Max_Corr|>0.1). Returns empty arrays if missing."""
    path = os.path.join(DATA_ROOT, _model_dir(name), "formant_dim_analysis_cache",
                        "Formant_Lag_Analysis_Results_moreLayers.csv")
    mc, bl = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["Layer"]) == layer and abs(float(row["Max_Corr"])) > 0.1:
                        mc.append(float(row["Max_Corr"]))
                        bl.append(float(row["Best_Lag_Sec"]))
                except (ValueError, TypeError, KeyError):
                    continue
    except Exception:
        return np.array([]), np.array([])
    return np.array(mc), np.array(bl)


def load_id(name, layer):
    """Per-language Intrinsic_Dim (only *_fleurs). Returns dict lang->ID."""
    path = os.path.join(DATA_ROOT, _model_dir(name), "ID_result",
                        f"ID_Analysis_L{layer}_Fixed.csv")
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lang = str(row.get("Language", "")).strip()
                if not lang.endswith("_fleurs"):
                    continue          # Exclude GLOBAL_MEAN / *_mls / _hn_rucheng
                try:
                    out[lang] = float(row["Intrinsic_Dim"])
                except (ValueError, TypeError):
                    continue
    except Exception:
        return {}
    return out


def load_area_table():
    """{(name, layer): (area, norm)}; model name matching is case-insensitive."""
    tbl = {}
    path = os.path.join(DATA_ROOT, "manifold_area.csv")
    if not os.path.exists(path):
        return tbl
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = str(row.get("Model", "")).strip()
            key = next((k for k in NAMES if k.lower() == m.lower()), None)
            if key is None:
                continue
            try:
                lv = int(row["Layer"])
                tbl[(key, lv)] = (float(row["Physical Coverage Area"]),
                                  float(row["Normalized Manifold Area"]))
            except (ValueError, TypeError, KeyError):
                continue
    return tbl


def model_total_layers():
    """Detect total number of layers from Formant CSV; fallback to 25 if missing."""
    totals = {}
    for name, d in MODELS:
        path = os.path.join(DATA_ROOT, d, "formant_dim_analysis_cache",
                            "Formant_Lag_Analysis_Results_moreLayers.csv")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                layers = [int(r["Layer"]) for r in csv.DictReader(f)]
            totals[name] = max(layers) + 1 if layers else 25
        except Exception:
            totals[name] = 25
    return totals


def detect_dataset(n_langs):
    if n_langs >= 20:
        return f"FLEURS-like ({n_langs} languages)"
    if n_langs > 0:
        return f"MLS-like ({n_langs} languages)"
    return "UNKNOWN (no data)"


# ---------------------------------------------------------------------------
# Round Layer Indexing
# ---------------------------------------------------------------------------
def round_layer(name, k, loops, totals):
    T = totals.get(name, 25)
    if T <= 0:
        return 0
    return max(0, min(int(round(k / loops * (T - 1))), T - 1))


FORCE_NONPARAM = False   # When True, all metrics are fixed to use non-parametric tests (for reproducibility across environments)


def effect_report(groups, label, indent="    "):
    """Perform normality test + hypothesis test + effect size + CI for a single metric, returning a summary string."""
    a, b = groups
    if len(a) == 0 or len(b) == 0:
        return f"{indent}{label:<14s} insufficient samples (n={len(a)}/{len(b)})"
    wa, pa = shapiro_wilk_pvalue(a)
    wb, pb = shapiro_wilk_pvalue(b)
    n1, n2 = len(a), len(b)
    normal = (not FORCE_NONPARAM and pa is not None and pb is not None
              and pa > 0.05 and pb > 0.05)
    if normal:
        t, p = welch_ttest(a, b)
        d = cohen_d(a, b)
        lo, hi = ci_cohen_d(d, n1, n2)
        if p is None:
            return f"{indent}{label:<14s} Welch t=n/a  d={d:+.2f} [{lo:+.2f},{hi:+.2f}]"
        star = "*" if p < 0.05 else " "
        return (f"{indent}{label:<14s} Welch t={t:.3f} p={fmt_p(p)} "
                f"d={d:+.2f} [{lo:+.2f},{hi:+.2f}]{star}")
    U, p = mann_whitney_u(a, b)
    cd = cliffs_delta(a, b)
    lo, hi = ci_cliffs(cd, n1, n2)
    if U is None:
        return f"{indent}{label:<14s} MW=n/a  Δ={cd:+.3f} [{lo:+.2f},{hi:+.2f}]"
    star = "*" if (p is not None and p < 0.05) else " "
    return (f"{indent}{label:<14s} MW U={U:.0f} p={fmt_p(p)} "
            f"Δ={cd:+.3f} [{lo:+.2f},{hi:+.2f}]{star}")


# ---------------------------------------------------------------------------
# Single Round Execution
# ---------------------------------------------------------------------------
def run_round(k, loops, totals, area_tbl, lang_floor, out):
    def p(s=""):
        out.append(s)
    frac = k / loops
    layers = {n: round_layer(n, k, loops, totals) for n in NAMES}

    p("\n" + "=" * 78)
    p(f" ROUND {k}/{loops} — Relative Depth {frac:.3f}")
    p("  Layer Index: " + " / ".join(f"{n.split('-')[-1]}={layers[n]}" for n in NAMES))
    p("=" * 78)

    # ---------- Data Collection ----------
    pc = {}      # name -> (mc_signed, bl)  —— Convex hull/area must use signed values (same meaning as manifold_area.csv)
    mca = {}     # name -> |Max_Corr|        —— Used only for Max_Corr distribution tests
    ids = {}     # name -> dict lang->ID
    areas = {}   # name -> (area, norm)
    for n in NAMES:
        mc, bl = load_point_cloud(n, layers[n])
        pc[n] = (mc, bl)
        mca[n] = np.abs(mc) if len(mc) else mc
        ids[n] = load_id(n, layers[n])
        areas[n] = area_tbl.get((n, layers[n]), (None, None))

    # Align ID per language (take languages common to all four models to ensure comparability)
    common = set(lang_floor)
    for n in NAMES:
        common &= set(ids[n].keys())
    common = sorted(common)
    id_vec = {n: np.array([ids[n][l] for l in common]) for n in NAMES}

    # ---------- Aggregated Metrics (Descriptive) ----------
    p("\n  Aggregated Metrics (Single Value/Model Layer — Descriptive):")
    hdr = (f"    {'Model':<12s}{'Layer':>6s}{'Params':>7s}{'Nodes':>7s}"
           f"{'ID(mean)':>10s}{'Area':>9s}{'NormArea':>10s}")
    p(hdr)
    for n in NAMES:
        mc = pc[n][0]; a, nm = areas[n]
        idm = id_vec[n].mean() if len(id_vec[n]) else float("nan")
        p(f"    {n:<12s}{layers[n]:>6d}{PARAMS_M[n]:>7d}{len(mc):>7d}"
          f"{(f'{idm:.2f}' if np.isfinite(idm) else 'n/a'):>10s}"
          f"{(f'{a:.4f}' if a is not None else 'n/a'):>9s}"
          f"{(f'{nm:.4f}' if nm is not None else 'n/a'):>10s}")

    # ---------- Paired Tests ----------
    for pair, kind in [(LANG_PAIR, "Same params / different language count (non-scaling)"),
                       (SCALING_PAIR, "Same language count / different params (true scaling)")]:
        a, b = pair
        p(f"\n  Pairwise {a} vs {b} [{kind}] (Layer {layers[a]}/{layers[b]}):")
        p(effect_report((mca[a], mca[b]), "Max_Corr"))
        p(effect_report((pc[a][1], pc[b][1]), "Best_Lag_Sec"))
        p(effect_report((id_vec[a], id_vec[b]), "Intrinsic_Dim"))

        # Area: bootstrap difference test
        # Convex hull area is a geometric quantity of signed (Max_Corr, Best_Lag), consistent with manifold_area.csv definition
        pa_pts = np.column_stack([pc[a][0], pc[a][1]]) if len(pc[a][0]) else np.zeros((0, 2))
        pb_pts = np.column_stack([pc[b][0], pc[b][1]]) if len(pc[b][0]) else np.zeros((0, 2))
        d, lo, hi = bootstrap_area_diff(pa_pts, pb_pts, seed=k)
        if np.isfinite(d):
            sig = "YES" if (lo > 0 or hi < 0) else " no"
            p(f"    {'Area(diff)':<14s} ΔA={d:+.4f} 95%CI [{lo:+.4f},{hi:+.4f}] excl0={sig}")
        else:
            p(f"    {'Area(diff)':<14s} insufficient samples")

    # ---------- Comprehensive Four-Model Test ----------
    p("\n  Comprehensive Four-Model (Omnibus):")
    for label, getter in [("Max_Corr", lambda n: mca[n]),
                          ("Best_Lag_Sec", lambda n: pc[n][1]),
                          ("Intrinsic_Dim", lambda n: id_vec[n])]:
        groups = [getter(n) for n in NAMES]
        nonempty = [g for g in groups if len(g) > 0]
        means = "  ".join(f"{n.split('-')[-1]}={np.mean(g):.3f}" if len(g) else f"{n.split('-')[-1]}=n/a"
                          for n, g in zip(NAMES, groups))
        p(f"    {label:<14s} {means}")
        if len(nonempty) < 2:
            p(f"      -> insufficient groups")
            continue
        normal = all(len(g) >= 3 and shapiro_wilk_pvalue(g)[1] is not None
                     and shapiro_wilk_pvalue(g)[1] > 0.05 for g in nonempty)
        if normal and sp_stats is not None:
            F, pv = anova_ftest(groups)
            p(f"      ANOVA F={F:.3f} p={fmt_p(pv)}")
        else:
            H, pv = kruskal_wallis_pvalue(groups)
            if H is None:
                p("      Kruskal-Wallis n/a (scipy unavailable)")
            else:
                p(f"      Kruskal-Wallis H={H:.3f} p={fmt_p(pv)}")

    # ---------- Scaling Law (Params vs Aggregated Metrics, n=4, exact df=2) ----------
    p("\n  Scaling Law (Pearson, n=4, exact df=2; |r|>=0.95 required for p<0.05):")
    par = np.array([PARAMS_M[n] for n in NAMES], float)
    metrics = {
        "Nodes":    np.array([len(pc[n][0]) for n in NAMES], float),
        "ID(mean)": np.array([id_vec[n].mean() if len(id_vec[n]) else np.nan for n in NAMES]),
        "Area":     np.array([areas[n][0] if areas[n][0] is not None else np.nan for n in NAMES]),
        "NormArea": np.array([areas[n][1] if areas[n][1] is not None else np.nan for n in NAMES]),
    }
    for lab, vals in metrics.items():
        if np.any(np.isnan(vals)):
            p(f"    {lab:<10s} n/a (missing values)")
            continue
        r, pv = pearson_exact(par, vals)
        p(f"    {lab:<10s} r={r:+.3f} p={fmt_p(pv)}")

    return {"k": k, "layers": dict(layers),
            "nodes": {n: len(pc[n][0]) for n in NAMES},
            "id_mean": {n: float(id_vec[n].mean()) if len(id_vec[n]) else None for n in NAMES},
            "area": {n: areas[n][0] for n in NAMES},
            "norm": {n: areas[n][1] for n in NAMES}}


# ---------------------------------------------------------------------------
class _Tee:
    """Writes simultaneously to screen and log file."""
    def __init__(self, path):
        self._terminal = sys.stdout
        self._log = open(path, 'w', encoding='utf-8', buffering=1)

    def write(self, m):
        self._terminal.write(m)
        self._log.write(m)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def isatty(self):
        return False

    def close(self):
        self._log.close()
        sys.stdout = self._terminal


def main():
    global DATA_ROOT

    global FORCE_NONPARAM
    loops = 8
    use_log = True
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--loops" and i + 1 < len(args):
            loops = max(2, int(args[i + 1])); i += 2
        elif args[i] == "--test" and i + 1 < len(args):
            mode = args[i + 1].lower()
            if mode in ("nonparam", "nonparametric", "mw"):
                FORCE_NONPARAM = True
            elif mode in ("auto", "default"):
                FORCE_NONPARAM = False
            else:
                print(f"[ERROR] --test only accepts auto | nonparam", file=sys.stderr)
                return 2
            i += 2
        elif args[i] == "--no-log":
            use_log = False; i += 1
        elif args[i] in ("--root",):
            # Explicitly disabled: this script only analyzes FLEURS
            print("[ERROR] This script is fixed to use the FLEURS directory re-organized/, --root is not supported.",
                  file=sys.stderr)
            return 2
        else:
            i += 1

    DATA_ROOT = pick_root()

    log_path = None
    if use_log:
        log_path = os.path.splitext(os.path.abspath(__file__))[0] + '.log'
        sys.stdout = _Tee(log_path)

    out = []
    def p(s=""):
        out.append(s)

    p("=" * 78)
    p(" Cyclic Depth Statistical Test — FLEURS (24 languages) · Six Metrics")
    p("   Max_Corr / Best_Lag_Sec / Intrinsic_Dim / Nodes / Area / NormArea")
    p(f" Data Root Directory: {DATA_ROOT}")
    p(f" Rounds: {loops}   Relative Depth per round = k/{loops}")
    p(f" scipy: {'available' if sp_stats is not None else 'MISSING'}"
      f"   Test Method: {'forced nonparametric' if FORCE_NONPARAM else 'auto (Shapiro->Welch/MW)'}")
    p("=" * 78)
    if sp_stats is None and not FORCE_NONPARAM:
        p("\n[WARNING] scipy not installed: normality tests cannot be performed, all metrics will default to non-parametric tests")
        p("          (Mann-Whitney + Cliff's Δ). Machines with scipy in auto mode will switch to Welch t + Cohen's d for normally distributed metrics,")
        p("          so effect sizes on both sides are not directly comparable.")
        p("          Add --test nonparam if cross-environment reproducibility is needed (force non-parametric on both sides).")

    if not os.path.isdir(DATA_ROOT):
        p(f"\n[ERROR] FLEURS data directory not found: {DATA_ROOT}")
        p(f"        Please place FLEURS data in {FLEURS_ROOT_NAME}/ at the same level as this script, with structure:")
        p(f"          {FLEURS_ROOT_NAME}/<model>/formant_dim_analysis_cache/"
          f"Formant_Lag_Analysis_Results_moreLayers.csv")
        p(f"          {FLEURS_ROOT_NAME}/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv")
        p(f"          {FLEURS_ROOT_NAME}/manifold_area.csv")
        _flush(out); return 1

    totals = model_total_layers()
    area_tbl = load_area_table()

    # ---------- Startup Diagnostics ----------
    p("\n[Diagnostic] Data Integrity Check")
    missing = []
    for n, d in MODELS:
        pc_path = os.path.join(DATA_ROOT, d, "formant_dim_analysis_cache",
                               "Formant_Lag_Analysis_Results_moreLayers.csv")
        id_dir = os.path.join(DATA_ROOT, d, "ID_result")
        ok_pc = os.path.exists(pc_path)
        ok_id = os.path.isdir(id_dir)
        n_id = len([f for f in os.listdir(id_dir) if f.startswith('ID_Analysis_L')]) if ok_id else 0
        if not ok_pc:
            missing.append(f"{n}:Formant CSV")
        if not ok_id:
            missing.append(f"{n}:ID_result")
        p(f"  {n:<12} layers={totals[n]:>3}  Formant={'OK' if ok_pc else 'MISSING'}"
          f"  ID_files={n_id:>3}  area_rows={sum(1 for kk in area_tbl if kk[0]==n):>3}")
    no_pc = [m for m in missing if m.endswith(':Formant CSV')]
    no_id = [m for m in missing if m.endswith(':ID_result')]
    if no_pc:
        p(f"\n[ERROR] Formant CSV missing (all metrics for these models will be empty): {', '.join(x.split(':')[0] for x in no_pc)}")
    if no_id:
        p(f"\n[INFO] The following models lack ID_result yet; ID tests will be automatically skipped: "
          f"{', '.join(x.split(':')[0] for x in no_id)}")
        p("       If ID is still being computed, this is expected behavior; only Max_Corr / Best_Lag / Nodes / Area participate in statistics.")
        p(f"       Once generated, place it at {DATA_ROOT}/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv.")
    if not missing:
        p("  -> Formant CSV and ID_result found for all four models")

    # Dataset identity verification (language count)
    probe = load_point_cloud(NAMES[0], 0)
    langs_pc = set()
    pc_path = os.path.join(DATA_ROOT, _model_dir(NAMES[0]), "formant_dim_analysis_cache",
                           "Formant_Lag_Analysis_Results_moreLayers.csv")
    if os.path.exists(pc_path):
        with open(pc_path, newline="", encoding="utf-8") as f:
            langs_pc = {str(r.get("Language", "")).strip() for r in csv.DictReader(f)}
    langs_pc.discard("")
    p(f"\n[Dataset] {detect_dataset(len(langs_pc))}  (distinct Language = {len(langs_pc)})")
    if len(langs_pc) == 0:
        p("  [WARNING] Could not read language names, please verify path and file format")
    elif len(langs_pc) < 20:
        p(f"  [ERROR] Language count {len(langs_pc)} < 20, this is not FLEURS (24 languages) data.")
        p("          This script is only for FLEURS; please check if re-organized/ contains FLEURS results.")
        _flush(out); return 3

    id_probe = load_id(NAMES[0], 0)
    lang_floor = sorted(id_probe.keys())
    if lang_floor:
        p(f"[ID] Model {NAMES[0]} Layer 0 *_fleurs language count = {len(lang_floor)}")
    else:
        p("[ID] No ID data detected -> All Intrinsic_Dim related tests skipped (other metrics normal)")

    # ---------- Per Round Execution ----------
    summary = []
    for k in range(1, loops + 1):
        summary.append(run_round(k, loops, totals, area_tbl, lang_floor, out))
        p("")
        _flush(out)

    # ---------- Cross-Round Summary ----------
    p("\n" + "=" * 78)
    p(" Cross-Round Summary — Changes in Aggregated Metrics with Relative Depth")
    p("-" * 78)
    p(f"{'R':<3}{'layers(53/300M/1B/2B)':<22}{'Nodes':<20}{'ID mean':<24}{'Area(1B/2B)':<18}")
    for s in summary:
        lay = "/".join(str(s['layers'][n]) for n in NAMES)
        nod = "/".join(str(s['nodes'][n]) for n in NAMES)
        idm = "/".join(f"{s['id_mean'][n]:.1f}" if s['id_mean'][n] is not None else "n/a" for n in NAMES)
        a1 = s['area'].get(SCALING_PAIR[0]); a2 = s['area'].get(SCALING_PAIR[1])
        ar = f"{a1:.3f}/{a2:.3f}" if (a1 is not None and a2 is not None) else "n/a"
        p(f"{s['k']:<3}{lay:<22}{nod:<20}{idm:<24}{ar:<18}")

    p("\n" + "=" * 78)
    p(" Notes")
    p("-" * 78)
    p(" * Max_Corr / Best_Lag: Per-node (|Max_Corr|>0.1) tests, significance inference possible")
    p(" * Intrinsic_Dim: Per-language (languages common to all four models ending in *_fleurs) tests, significance inference possible")
    p(" * Nodes: Number of significant dimensions per layer, single value -> descriptive only")
    p(" * Area / NormArea: Convex hull area per layer, single value -> bootstrap CI + difference bootstrap test")
    p(" * Scaling Law: Parameters vs Aggregated Metrics, Pearson exact t (n=4 -> df=2, critical |r|=0.95)")
    p(" * True scaling comparison = XLS-R-1B vs XLS-R-2B (same 128 languages / different parameters);")
    p("   XLSR-53 vs XLS-R-300M (same ~300M params / different language counts 53vs128) is a data scale effect, not scaling.")
    p("=" * 78)

    _flush(out)

    if use_log and isinstance(sys.stdout, _Tee):
        print(f"\n[Log] Saved: {log_path}", file=sys.__stdout__)
        sys.stdout.close()
    return 0


def _flush(out):
    """Write accumulated lines at once (facilitates tee to file)."""
    if not out:
        return
    sys.stdout.write("\n".join(out) + "\n")
    out.clear()


if __name__ == "__main__":
    sys.exit(main())
