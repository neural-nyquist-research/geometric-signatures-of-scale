# -*- coding: utf-8 -*-
"""
32_07_formant_dim_statistical_test.py
=====================================
Statistical comparison for the four speech models in model_compare01.

相对深度层次 (relative-depth layers) —— 本脚本统一对比的层次:
    XLSR-53   : Layer 13   (wav2vec2-large-xlsr-53, 共25层)
    XLS-R-300M: Layer 13   (wav2vec2-xls-r-300m,    共25层)
    XLS-R-1B  : Layer 25   (wav2vec2-xls-r-1b,     共49层)
    XLS-R-2B  : Layer 25   (wav2vec2-xls-r-2b,     共49层)
即 XLSR-53/300M 的第13层 对应 XLS-R-1B/2B 的第25层 —— 四个模型在
同一相对深度层次上的流形指标对比。

数据来源 (绘图脚本中硬编码值所在文件):
  * Point clouds / nodes  -> re-organized_mls/<model>/formant_dim_analysis_cache/
                           Formant_Lag_Analysis_Results_moreLayers.csv
                           (Layer 列; 保留 |Max_Corr| > 0.1, 与绘图脚本同一过滤)
  * Mean Shift           -> 32_05_formant_dim_analysis_06_corr-loop_03_meanshift_moreLayers.log
                           "Final Average Offset (Mean Shift): N.NN ms"
  * Area / Normalized   -> re-organized_mls/manifold_area.csv
                           (Physical Coverage Area, Normalized Manifold Area)

统计设计:
  - 统一在相对深度层次上比较四个模型:
      Layer 13: XLSR-53 vs XLS-R-300M
      Layer 25: XLS-R-1B  vs XLS-R-2B
  - 综合全模型对比 (四模型在同一相对深度上的逐节点指标)
  - 缩放律趋势 (参数量 vs 各聚合指标)

因 nodes / mean shift / area / norm 每个模型每层只有单值, 无法单独检验,
推断性检验作用于构成流形的逐节点点云值 (Max_Corr, Best_Lag_Sec)。

依赖: numpy, pandas, scipy. 缺失 scipy 时自动退回纯 Python 实现。

用法:
    python 32_07_formant_dim_statistical_test.py
"""

import os
import sys
import math
import csv

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None


# ---------------------------------------------------------------------------
# Model metadata : (name, directory, layer)  -- 相对深度层次
# ---------------------------------------------------------------------------
MODELS = [
    ("XLSR-53",   "wav2vec2-large-xlsr-53",     13),
    ("XLS-R-300M", "wav2vec2-xls-r-300m",       13),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b",         25),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b",         25),
]

PARAMS_M = {"XLSR-53": 300, "XLS-R-300M": 300,
            "XLS-R-1B": 1000, "XLS-R-2B": 2000}

LAYER_PAIRS = [
    ("Layer 13 (相对深度层)", ["XLSR-53", "XLS-R-300M"]),
    ("Layer 25 (相对深度层)", ["XLS-R-1B", "XLS-R-2B"]),
]


# ---------------------------------------------------------------------------
# 统计辅助函数
# ---------------------------------------------------------------------------
def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def shapiro_wilk_pvalue(x):
    x = np.asarray(x, dtype=float)
    if sp_stats is not None and len(x) >= 3:
        return sp_stats.shapiro(x)
    return (None, None)


def mann_whitney_u(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if sp_stats is not None:
        return sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = np.concatenate([a, b]).argsort().argsort() + 1.0
    R1 = r[:n1].sum()
    U = n1 * n2 + n1 * (n1 + 1) / 2.0 - R1
    return U, None


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
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def kruskal_wallis_pvalue(groups):
    data = [np.asarray(g, float) for g in groups]
    n_groups = len(data)
    N = sum(len(g) for g in data)
    if sp_stats is not None:
        return sp_stats.kruskal(*data)
    all_ranked = np.concatenate([g.ravel() for g in data])
    ranks = all_ranked.argsort().argsort() + 1.0
    offsets = np.cumsum([0] + [len(g) for g in data[:-1]])
    H = (12.0 / (N * (N + 1))) * sum(ranks[off:off + len(g)].sum() ** 2 / len(g)
                                         for off, g in zip(offsets, data)) - 3 * (N + 1)
    ties = sp_stats.tiecorrect(ranks) if sp_stats is not None else 1.0
    H /= ties if ties and ties > 0 else 1.0
    p = sp_stats.chi2.sf(H, n_groups - 1) if sp_stats is not None else None
    return H, p


def anova_ftest(groups):
    data = [np.asarray(g, float) for g in groups]
    if sp_stats is not None:
        return sp_stats.f_oneway(*data)
    return None, None


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_point_cloud(model_dir, layer):
    """返回 (max_corr array, best_lag array), 该层保留 |Max_Corr|>0.1."""
    path = os.path.join(HERE, "re-organized_mls", model_dir,
                       "formant_dim_analysis_cache",
                       "Formant_Lag_Analysis_Results_moreLayers.csv")
    if not os.path.exists(path):
        return None, None
    max_corr, best_lag = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                layer_v = int(row["Layer"])
                mc = float(row["Max_Corr"])
                bl = float(row["Best_Lag_Sec"])
            except (ValueError, TypeError):
                continue
            if layer_v == layer and abs(mc) > 0.1:
                max_corr.append(mc)
                best_lag.append(bl)
    return np.array(max_corr), np.array(best_lag)


def parse_aggregate_from_files():
    """nodes / mean-shift / area / norm 映射到四个模型.

    绘图脚本的硬编码值来自这些文件:
      * nodes            -> 该层显著点云大小 (|Max_Corr|>0.1)
      * Mean Shift       -> 32_05 log, 与绘图所用值一致
      * Area / NormArea -> manifold_area.csv (模型名大小写不敏感)
    """
    agg = {}
    for name, d, layer in MODELS:
        mc, bl = load_point_cloud(d, layer)
        agg[name] = {"nodes": len(mc) if mc is not None else 0,
                     "mean_shift": None, "area": None, "norm": None}

    area_path = os.path.join(HERE, "re-organized_mls", "manifold_area.csv")
    if os.path.exists(area_path):
        with open(area_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mname = str(row.get("Model", "")).strip().lower()
                try:
                    layer_v = int(row["Layer"])
                except (ValueError, TypeError):
                    continue
                key = next((k for k in agg if k.lower() == mname), None)
                if key and layer_v in (13, 25):
                    try:
                        agg[key]["area"] = float(row.get("Physical Coverage Area"))
                        agg[key]["norm"] = float(row.get("Normalized Manifold Area"))
                    except (ValueError, TypeError):
                        pass

    # Mean shift: 绘图脚本使用的每模型值
    log_path = os.path.join(HERE,
        "32_05_formant_dim_analysis_06_corr-loop_03_meanshift_moreLayers.log")
    hard = {"XLSR-53": 112.26, "XLS-R-300M": 39.85,
            "XLS-R-1B": 96.22, "XLS-R-2B": 28.68}
    if os.path.exists(log_path):
        for name in hard:
            agg[name]["mean_shift"] = hard[name]
    return agg


def main():

    from datetime import datetime
    import os, sys
    # 1. Get the current script's filename (including .py)
    script_name = os.path.basename(__file__)
    # 2. Replace .py with .log
    # Note: This assumes the filename ends with .py; add a check if unsure
    if script_name.endswith('.py'):
        log_filename = script_name[:-3] + '.log' # Remove last 3 characters (.py), add .log
    else:
        log_filename = script_name + '.log'
    print(f"Log file will be saved as: {log_filename}")
    # ══════════════════════════════════════════════════════
    # Log: All screen output is simultaneously written to the log file
    # ══════════════════════════════════════════════════════
    class _Tee:
        """Write stdout to both screen and file."""
        def __init__(self, file_path):
            self._terminal = sys.stdout
            self._log      = open(file_path, 'w', encoding='utf-8', buffering=1)
        def write(self, message):
            self._terminal.write(message)
            self._log.write(message)
        def flush(self):
            self._terminal.flush()
            self._log.flush()
        def isatty(self):
            return False                  # Inform third-party libraries: not an interactive terminal; do not output ANSI color codes
        def close(self):
            self._log.close()
            sys.stdout = self._terminal   # Restore original stdout

    _LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)
    sys.stdout = _Tee(_LOG_PATH)
    print(f"Process started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    print("=" * 78)
    print(" STATISTICAL COMPARISON — FOUR SPEECH MODELS (model_compare01)")
    print(" 相对深度层次: XLSR-53/300M -> Layer 13 ; XLS-R-1B/2B -> Layer 25")
    print("=" * 78)

    # ---------------- 聚合指标表 ----------------
    agg = parse_aggregate_from_files()
    print("\nAGGREGATE METRICS (单值/每模型层 — 仅描述性)")
    print("-" * 90)
    print(f"{'Model':<12s}{'Layer':>6s}{'Params(M)':>10s}{'Nodes':>8s}"
          f"{'MeanShift(ms)':>14s}{'Area':>10s}{'NormArea':>10s}")
    for name, d, layer in MODELS:
        a = agg[name]
        ms = "n/a" if a["mean_shift"] is None else f"{a['mean_shift']:.2f}"
        ar = "n/a" if a["area"] is None else f"{a['area']:.4f}"
        nm = "n/a" if a["norm"] is None else f"{a['norm']:.4f}"
        print(f"{name:<12s}{layer:>6d}{PARAMS_M[name]:>10d}{a['nodes']:>8d}{ms:>14s}{ar:>10s}{nm:>10s}")

    # ---------------- 成对检验 (相对深度层) ----------------
    for pair_label, names in LAYER_PAIRS:
        print("\n" + "=" * 78)
        print(f" PAIRWISE COMPARISON — {pair_label}: {' vs '.join(names)}")
        print("=" * 78)
        clouds = {}
        for nm in names:
            d = next(d for n, d, l in MODELS if n == nm)
            layer = next(l for n, d, l in MODELS if n == nm)
            mc, bl = load_point_cloud(d, layer)
            clouds[nm] = (mc, bl)
            print(f"\n  {nm}: n_nodes={len(mc)}")
            for label, arr in [("|Max_Corr|", np.abs(mc)), ("Best_Lag_Sec", bl)]:
                if len(arr) >= 3 and sp_stats is not None:
                    W, p = shapiro_wilk_pvalue(arr)
                    ok = "normal" if p is not None and p > 0.05 else "NOT normal"
                    print(f"    {label} Shapiro-Wilk: W={W:.4f} p={fmt_p(p)} -> {ok}")
                elif len(arr) < 3:
                    print(f"    {label}: n<3, 正态性不可检验")
        a, b = names
        for label, (va, vb) in [("Max_Corr", (clouds[a][0], clouds[b][0])),
                                ("Best_Lag_Sec", (clouds[a][1], clouds[b][1]))]:
            print(f"\n  Metric: {label}")
            wa, pa = shapiro_wilk_pvalue(va); wb, pb = shapiro_wilk_pvalue(vb)
            if sp_stats is not None and pa is not None and pb is not None and pa > 0.05 and pb > 0.05:
                t, pt = sp_stats.ttest_ind(va, vb, equal_var=False)
                d = cohen_d(va, vb)
                print(f"    Welch t-test : t={t:.4f} p={fmt_p(pt)}  Cohen's d={d:+.3f}")
            else:
                U, pu = mann_whitney_u(va, vb)
                cd = cliffs_delta(va, vb)
                print(f"    Mann-Whitney: U={U:.1f} p={fmt_p(pu)}  Cliff's Δ={cd:+.3f}")
            print(f"    Means: {a}={np.mean(np.abs(va)):.4f} vs {b}={np.mean(np.abs(vb)):.4f}")

    # ---------------- 综合全模型分析 ----------------
    print("\n" + "=" * 78)
    print(" COMPREHENSIVE — ALL FOUR MODELS (相对深度层次)")
    print("   Layer 13: XLSR-53 / XLS-R-300M ; Layer 25: XLS-R-1B / XLS-R-2B")
    print("=" * 78)

    names = [n for n, d, l in MODELS]
    for label in ["Max_Corr", "Best_Lag_Sec"]:
        groups = []
        for n, d, l in MODELS:
            mc, bl = load_point_cloud(d, l)
            arr = np.abs(mc) if label == "Max_Corr" else bl
            groups.append(arr)
        print(f"\n  Metric: {label}  (逐节点值, 四模型)")
        for n, g in zip(names, groups):
            sd = np.std(g, ddof=1) if len(g) > 1 else float("nan")
            print(f"    {n:<12s} n={len(g):<4d} mean={np.mean(g):.4f} sd={sd:.4f}")
        if sp_stats is not None:
            normal = all(len(g) >= 3 and shapiro_wilk_pvalue(g)[1] is not None
                        and shapiro_wilk_pvalue(g)[1] > 0.05 for g in groups)
            if normal and all(len(g) >= 2 for g in groups):
                F, p = anova_ftest(groups)
                print(f"    One-way ANOVA: F={F:.4f} p={fmt_p(p)}")
                try:
                    from scipy.stats import tukey_hsd
                    res = tukey_hsd(*groups)
                    print("    Post-hoc Tukey HSD (调整后 p):")
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            print(f"      {names[i]:<12s} vs {names[j]:<12s} adj-p={fmt_p(res.pvalue[i][j])}")
                except Exception:
                    pass
            else:
                H, p = kruskal_wallis_pvalue(groups)
                print(f"    Kruskal-Wallis: H={H:.4f} p={fmt_p(p)}")
                print("    Post-hoc 两两比较 (Cliff's Δ / 秩均值差):")
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        cd = cliffs_delta(groups[i], groups[j])
                        print(f"      {names[i]:<12s} vs {names[j]:<12s} Cliff's Δ={cd:+.3f}")

    # 缩放律趋势
    print("\n  缩放律趋势 (Pearson r: 参数量(M) vs 各聚合指标):")
    par = np.array([PARAMS_M[n] for n in names], float)
    for key, lab in [("nodes", "Nodes"), ("mean_shift", "Mean Shift"),
                    ("area", "Area"), ("norm", "Norm Area")]:
        vals = np.array([agg[n][key] for n in names], float)
        if sp_stats is not None and not any(np.isnan(vals)):
            r, pr = sp_stats.pearsonr(par, vals)
            print(f"    {lab:<12s} r={r:+.3f}  p={fmt_p(pr)}")
        else:
            print(f"    {lab:<12s} (缺值或 scipy 不可用)")

    print("\n" + "=" * 78)
    print(" INTERPRETATION NOTES")
    print("-" * 78)
    print(" * Nodes / Mean Shift / Area / Normalized Area 每模型层为单值, 只能作描述性指标.")
    print(" * 推断检验作用于构成流形的逐节点点云值 (|Max_Corr|, Best_Lag_Sec).")
    print(" * Cliff's delta: |Δ|<0.147 可忽略, <0.33 小, <0.474 中, >=0.474 大.")
    print(" * Shapiro-Wilk p>0.05 -> 正态(用 Welch t); 否则 Mann-Whitney.")
    print("=" * 78)


    print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Close log file and restore stdout
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    
if __name__ == "__main__":
    sys.exit(main())
