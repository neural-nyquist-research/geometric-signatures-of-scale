# -*- coding: utf-8 -*-
"""
02_07_formant_dim_statistical_test_comprehensive.py
=================================================
循环深度检验 — 四个语音模型在 MLS/FLEURS 数据集上的统计比较。

核心思想:
  对每个模型取「相对深度」层次做统计检验。默认 loops=8 轮,
  第 k 轮 (k = 1..loops) 对应的层索引为:
      layer_k(model) = int(round(k / loops * (T_model - 1)))   (0-indexed, 夹取到 [0, T-1])
  其中 T_model 为该模型的总层数 (由 CSV 中出现的最大 Layer 值 +1 决定):
      XLSR-53=25, XLS-R-300M=25, XLS-R-1B=49, XLS-R-2B=49
  因此:
      loops=8 -> 轮次深度分数: 1/8, 2/8, ..., 8/8
                 (XLSR-53/300M: 层 3,6,9,12,15,18,21,24 ; XLS-R-1B/2B: 层 6,12,18,24,30,36,42,48)
  每轮输出:
      1) 聚合指标表 (nodes / mean_shift / area / norm — 描述性单值)
      2) 成对检验: XLSR-53 vs XLS-R-300M (同深度层) 与 XLS-R-1B vs XLS-R-2B
      3) 综合全模型 Kruskal-Wallis/ANOVA + 事后比较
      4) 缩放律 Pearson r
  最后汇总: 各轮结论随深度的变化趋势。

用法:
    python 02_07_formant_dim_statistical_test_comprehensive.py            # 默认 8 轮
    python 02_07_formant_dim_statistical_test_comprehensive.py --loops 12 # 自定义轮次

依赖: numpy, pandas, scipy (缺失时自动退回纯 Python).
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
# 模型元数据 : (名字, 目录) — 总层数由 CSV 自动检测
# ---------------------------------------------------------------------------
MODELS = [
    ("XLSR-53",   "wav2vec2-large-xlsr-53"),
    ("XLS-R-300M", "wav2vec2-xls-r-300m"),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b"),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b"),
]

PARAMS_M = {"XLSR-53": 300, "XLS-R-300M": 300,
            "XLS-R-1B": 1000, "XLS-R-2B": 2000}

PAIR_GROUPS = [("XLSR-53", "XLS-R-300M"), ("XLS-R-1B", "XLS-R-2B")]


def model_total_layers():
    """从 CSV 检测每个模型总层数 (max Layer + 1)。"""
    totals = {}
    for name, d in MODELS:
        path = os.path.join(HERE, "re-organized_mls", d,
                          "formant_dim_analysis_cache",
                          "Formant_Lag_Analysis_Results_moreLayers.csv")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            layers = [int(r["Layer"]) for r in rows]
            totals[name] = max(layers) + 1 if layers else 0
        except Exception:
            totals[name] = 25
    return totals


TOTAL_LAYERS = model_total_layers()


# ---------------------------------------------------------------------------
# 统计辅助函数 (对空/短样本鲁棒)
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
    if len(a) == 0 or len(b) == 0:
        return None, None
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
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def kruskal_wallis_pvalue(groups):
    data = [np.asarray(g, float) for g in groups]
    data = [g for g in data if len(g) > 0]
    n_groups = len(data)
    if n_groups < 2:
        return None, None
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
    data = [g for g in data if len(g) > 0]
    if len(data) < 2:
        return None, None
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


def parse_aggregate_from_files(layers_map=None):
    """nodes / mean-shift / area / norm 映射到四个模型 (与绘图脚本硬编码一致).

    若给定 layers_map={name:layer}, 则 nodes 按该层计算; 否则用最深一层。
    """
    agg = {}
    for name, d in MODELS:
        T = TOTAL_LAYERS.get(name, 25)
        layer = layers_map.get(name) if layers_map else None
        if layer is None:
            layer = max(0, T - 1)
        mc, bl = load_point_cloud(d, layer)
        agg[name] = {"layer": layer,
                     "nodes": len(mc) if mc is not None else 0,
                     "mean_shift": None, "area": None, "norm": None}

    area_path = os.path.join(HERE, "re-organized_mls", "manifold_area.csv")
    if os.path.exists(area_path):
        with open(area_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mname = str(row.get("Model", "")).strip().lower()
                key = next((k for k in agg if k.lower() == mname), None)
                if key and agg[key]["area"] is None:
                    try:
                        agg[key]["area"] = float(row.get("Physical Coverage Area"))
                        agg[key]["norm"] = float(row.get("Normalized Manifold Area"))
                    except (ValueError, TypeError):
                        pass

    log_path = os.path.join(HERE,
        "32_05_formant_dim_analysis_06_corr-loop_03_meanshift_moreLayers.log")
    hard = {"XLSR-53": 112.26, "XLS-R-300M": 39.85,
            "XLS-R-1B": 96.22, "XLS-R-2B": 28.68}
    if os.path.exists(log_path):
        for name in hard:
            agg[name]["mean_shift"] = hard[name]
    return agg


def round_layer(name, k, loops):
    """第 k 轮 (1..loops) 的相对深度层索引: int(round(k/loops*(T-1)))."""
    T = TOTAL_LAYERS.get(name, 25)
    if T <= 0:
        return 0
    idx = int(round(k / loops * (T - 1)))
    return max(0, min(idx, T - 1))


def run_round(k, loops):
    """执行第 k 轮检验."""
    frac = k / loops
    names = [n for n, d in MODELS]
    layers = {n: round_layer(n, k, loops) for n in names}

    print("\n" + "=" * 78)
    print(f" ROUND {k}/{loops} — 相对深度 {frac:.3f}")
    print("  层索引: XLSR-53={} / 300M={} / 1B={} / 2B={}".format(
        layers["XLSR-53"], layers["XLS-R-300M"],
        layers["XLS-R-1B"], layers["XLS-R-2B"]))
    print("=" * 78)

    agg = {}
    for n in names:
        mc, bl = load_point_cloud(next(d for nn, d in MODELS if nn == n), layers[n])
        agg[n] = {"nodes": len(mc) if mc is not None else 0}
    print("\n  聚合指标 (单值/模型层 — 描述性):")
    print(f"    {'Model':<12s}{'Layer':>6s}{'Params(M)':>10s}{'Nodes':>8s}")
    for n in names:
        print(f"    {n:<12s}{layers[n]:>6d}{PARAMS_M[n]:>10d}{agg[n]['nodes']:>8d}")

    pair_results = {}
    for a, b in PAIR_GROUPS:
        da = next(d for nn, d in MODELS if nn == a)
        db = next(d for nn, d in MODELS if nn == b)
        va = load_point_cloud(da, layers[a])
        vb = load_point_cloud(db, layers[b])
        print(f"\n  成对 {a} vs {b}  (层 {layers[a]}/{layers[b]}):")
        for label, (x, y) in [("Max_Corr", (va[0], vb[0])),
                              ("Best_Lag_Sec", (va[1], vb[1]))]:
            x = np.abs(x) if label == "Max_Corr" else x
            y = np.abs(y) if label == "Max_Corr" else y
            wa, pa_ = shapiro_wilk_pvalue(x); wb, pb_ = shapiro_wilk_pvalue(y)
            line = f"    {label:<14s}"
            if len(x) == 0 or len(y) == 0:
                line += "样本不足, 跳过"
                print(line)
                pair_results[(a, b, label)] = (None, None)
                continue
            if sp_stats is not None and pa_ is not None and pb_ is not None \
               and pa_ > 0.05 and pb_ > 0.05:
                t, pt = sp_stats.ttest_ind(x, y, equal_var=False)
                d_eff = cohen_d(x, y)
                line += f"Welch t={t:.3f} p={fmt_p(pt)} d={d_eff:+.2f}"
            else:
                U, pu = mann_whitney_u(x, y)
                cd = cliffs_delta(x, y)
                line += f"MannWhit U={U if U is not None else float('nan'):.0f} p={fmt_p(pu)} Δ={cd:+.3f}"
            print(line)
            pair_results[(a, b, label)] = (pa_, pb_)

    names4 = [n for n, d in MODELS]
    for label in ["Max_Corr", "Best_Lag_Sec"]:
        groups = []
        for n in names4:
            d = next(dd for nn, dd in MODELS if nn == n)
            mc, bl = load_point_cloud(d, layers[n])
            arr = np.abs(mc) if label == "Max_Corr" else bl
            groups.append(arr)
        print(f"\n  综合 {label} (四模型, 逐节点):")
        for n, g in zip(names4, groups):
            sd = np.std(g, ddof=1) if len(g) > 1 else float("nan")
            mn = np.mean(g) if len(g) > 0 else float("nan")
            print(f"    {n:<12s} n={len(g):<4d} mean={mn:.4f} sd={sd:.4f}")
        if sp_stats is not None:
            nonempty = [g for g in groups if len(g) > 0]
            normal = (len(nonempty) == len(groups) and
                     all(len(g) >= 3 and shapiro_wilk_pvalue(g)[1] is not None
                         and shapiro_wilk_pvalue(g)[1] > 0.05 for g in nonempty))
            if normal and all(len(g) >= 2 for g in nonempty):
                F, p = anova_ftest(groups)
                print(f"    ANOVA: F={F:.4f} p={fmt_p(p)}")
                try:
                    from scipy.stats import tukey_hsd
                    res = tukey_hsd(*nonempty)
                    for i in range(len(nonempty)):
                        for j in range(i + 1, len(nonempty)):
                            print(f"      {names4[i]:<12s} vs {names4[j]:<12s} adj-p={fmt_p(res.pvalue[i][j])}")
                except Exception:
                    pass
            else:
                H, p = kruskal_wallis_pvalue(groups)
                print(f"    Kruskal-Wallis: H={H if H is not None else float('nan'):.4f} p={fmt_p(p)}")
                for i in range(len(names4)):
                    for j in range(i + 1, len(names4)):
                        cd = cliffs_delta(groups[i], groups[j])
                        print(f"      {names4[i]:<12s} vs {names4[j]:<12s} Cliff's Δ={cd:+.3f}")

    par = np.array([PARAMS_M[n] for n in names4], float)
    agg_full = parse_aggregate_from_files(layers_map=layers)
    print(f"\n  缩放律 (Pearson r: 参数量(M) vs 聚合指标, 层={{layers['XLSR-53']}}/{{layers['XLS-R-300M']}}/{{layers['XLS-R-1B']}}/{{layers['XLS-R-2B']}}):")
    for key, lab in [("nodes", "Nodes"), ("mean_shift", "Mean Shift"),
                    ("area", "Area"), ("norm", "Norm Area")]:
        vals = np.array([agg_full[n][key] for n in names4], float)
        if sp_stats is not None and not any(np.isnan(vals)):
            r, pr = sp_stats.pearsonr(par, vals)
            print(f"    {lab:<12s} r={r:+.3f}  p={fmt_p(pr)}")
        else:
            print(f"    {lab:<12s} (缺值或 scipy 不可用)")

    return {"round": k, "frac": frac, "layers": layers, "pair": pair_results}


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

    loops = 8
    if "--loops" in sys.argv:
        try:
            idx = sys.argv.index("--loops")
            loops = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            loops = 8
    if loops < 2:
        loops = 2

    print("=" * 78)
    print(f" 循环深度统计检验 — 四个语音模型 ({len(MODELS)} models)")
    print(f" 总层数: XLSR-53={TOTAL_LAYERS['XLSR-53']} / 300M={TOTAL_LAYERS['XLS-R-300M']} "
          f"/ 1B={TOTAL_LAYERS['XLS-R-1B']} / 2B={TOTAL_LAYERS['XLS-R-2B']}")
    print(f" 轮次: {loops}  每轮深度分数 = k/{loops}")
    print("=" * 78)

    summary = []
    for k in range(1, loops + 1):
        res = run_round(k, loops)
        summary.append(res)

    print("\n" + "=" * 78)
    print(" 跨轮汇总 — 成对检验 p 值随相对深度的变化")
    print("-" * 90)
    header = f"{'Round':<6s}{'frac':>6s}"
    for a, b in PAIR_GROUPS:
        header += f"{a[:8] + '|' + b[7:9]:>24s}".replace("XLS-R-", "")
    print(header)
    for res in summary:
        row = f"{res['round']:<6d}{res['frac']:.3f}"
        for a, b in PAIR_GROUPS:
            pa_, pb_ = res["pair"].get((a, b, "Max_Corr"), (None, None))
            row += f"{'p=' + fmt_p(pa_) if pa_ is not None else 'n/a':>24s}"
        print(row)

    print("\n" + "=" * 78)
    print(" INTERPRETATION NOTES")
    print("-" * 78)
    print(f" * 每轮层索引 = round(k/{loops} * (总层数-1))，故 53/300M 与 1B/2B 的层不同但相对深度相同。")
    print(" * nodes / mean shift / area / norm 为单值, 仅描述性; 推断检验作用于逐节点点云值。")
    print(" * Cliff's delta: |Δ|<0.147 可忽略, <0.33 小, <0.474 中, >=0.474 大。")
    print("=" * 78)

    print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Close log file and restore stdout
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()


if __name__ == "__main__":
    sys.exit(main())
