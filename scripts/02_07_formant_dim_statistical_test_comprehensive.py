# -*- coding: utf-8 -*-
"""
02_07_formant_dim_statistical_test_comprehensive.py
===================================================
四个 wav2vec2 系模型（XLSR-53 / XLS-R-300M / XLS-R-1B / XLS-R-2B）的
「相对深度」循环统计检验 —— 自包含版本，覆盖全部六类指标：

  1. Max_Corr       逐节点（逐维度）相关性幅度    —— 可做逐节点检验
  2. Best_Lag_Sec   逐节点最优时滞              —— 可做逐节点检验
  3. Intrinsic_Dim  逐语言内在维度 (ID)          —— 可做逐语言检验
  4. Nodes          每层显著维度数（单值）        —— 描述性
  5. Area           凸包物理面积（逐层）          —— bootstrap CI + 差值检验
  6. NormArea       归一化流形面积（逐层）        —— 同上

数据来源（仅 FLEURS 根目录 re-organized/；本脚本不读取 MLS 数据）:
  <root>/<model>/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv
         列 Language, Layer, Dimension, Formant, Max_Corr, Best_Lag_Sec
         nodes = 该层 |Max_Corr| > 0.1 的行数
  <root>/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv
         列 Language, Intrinsic_Dim ; 仅取 *_fleurs 语言 (排除 GLOBAL_MEAN 等)
  <root>/manifold_area.csv
         列 Model, Layer, Physical Coverage Area, Normalized Manifold Area
         (CSV 中 300M 写作 'XLS-R-300m'，按大小写不敏感匹配)

用法:
    python 02_07_formant_dim_statistical_test_comprehensive.py              # 默认 8 轮
    python 02_07_formant_dim_statistical_test_comprehensive.py --loops 12   # 自定义轮次
    python 02_07_formant_dim_statistical_test_comprehensive.py --no-log     # 不写日志文件

注意: 本脚本固定分析 FLEURS (24 语言) 数据, 不支持 --root, 也不会回退到 MLS 目录。

依赖: numpy 必需; scipy 可选（缺失时用纯 Python 实现，p 值可能为 n/a）。
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
# 模型元数据
# ---------------------------------------------------------------------------
MODELS = [
    ("XLSR-53",    "wav2vec2-large-xlsr-53"),
    ("XLS-R-300M", "wav2vec2-xls-r-300m"),
    ("XLS-R-1B",   "wav2vec2-xls-r-1b"),
    ("XLS-R-2B",   "wav2vec2-xls-r-2b"),
]
NAMES = [m[0] for m in MODELS]
PARAMS_M = {"XLSR-53": 300, "XLS-R-300M": 300, "XLS-R-1B": 1000, "XLS-R-2B": 2000}

# 真正的缩放对比（同 128 语言、不同参数）；另一对同参数、不同语言数
SCALING_PAIR = ("XLS-R-1B", "XLS-R-2B")
LANG_PAIR = ("XLSR-53", "XLS-R-300M")

# 本脚本仅供 FLEURS (24 语言) 数据分析使用: 数据根固定为 re-organized/
# 不读取也不回退到 MLS 目录 (re-organized_mls/)。
FLEURS_ROOT_NAME = "re-organized"
_ROOT_CANDIDATES = [FLEURS_ROOT_NAME]
DATA_ROOT = os.path.join(HERE, FLEURS_ROOT_NAME)


def pick_root():
    """返回 FLEURS 数据根目录 <script_dir>/re-organized/。

    不做任何 MLS 回退: 缺少该目录即视为配置错误, 由调用方报错退出。
    """
    return os.path.join(HERE, FLEURS_ROOT_NAME)


def count_languages(root):
    """返回首个模型 CSV 中的 distinct Language 数 (用于数据集身份校验)。"""
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
# 统计辅助函数（scipy 缺失时优雅退化）
# ---------------------------------------------------------------------------
def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _betainc_reg(a, b, x):
    """正则化不完全 Beta 函数（用于精确 t 分布 p 值）。"""
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
    """Pearson r + 精确双侧 p（Student t, df=n-2）。n=4 时 df=2，临界 |r|=0.95。"""
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
# 凸包面积 / bootstrap
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
    """点估计=全点云凸包; CI=bootstrap 百分位。"""
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    obs = hull_area(pts)
    rng = np.random.default_rng(seed)
    vals = np.array([hull_area(pts[rng.integers(0, len(pts), len(pts))])
                     for _ in range(n_boot)])
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_area_diff(a, b, n_boot=2000, seed=0):
    """两模型凸包面积差的 bootstrap: (obs_diff, lo, hi)。"""
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
# 数据加载
# ---------------------------------------------------------------------------
def _model_dir(name):
    return next(d for n, d in MODELS if n == name)


def load_point_cloud(name, layer):
    """逐节点 (|Max_Corr|>0.1) 的 Max_Corr / Best_Lag。缺失返回空数组。"""
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
    """逐语言 Intrinsic_Dim (仅 *_fleurs)。返回 dict lang->ID。"""
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
                    continue          # 排除 GLOBAL_MEAN / *_mls / _hn_rucheng
                try:
                    out[lang] = float(row["Intrinsic_Dim"])
                except (ValueError, TypeError):
                    continue
    except Exception:
        return {}
    return out


def load_area_table():
    """{(name, layer): (area, norm)}；模型名大小写不敏感。"""
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
    """从 Formant CSV 检测总层数；缺失回退 25。"""
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
# 轮次层索引
# ---------------------------------------------------------------------------
def round_layer(name, k, loops, totals):
    T = totals.get(name, 25)
    if T <= 0:
        return 0
    return max(0, min(int(round(k / loops * (T - 1))), T - 1))


FORCE_NONPARAM = False   # True 时对所有指标固定用非参数检验 (跨环境可复现)


def effect_report(groups, label, indent="    "):
    """对单指标做正态性 + 检验 + 效应量 + CI，返回一行摘要字符串。"""
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
# 单轮检验
# ---------------------------------------------------------------------------
def run_round(k, loops, totals, area_tbl, lang_floor, out):
    def p(s=""):
        out.append(s)
    frac = k / loops
    layers = {n: round_layer(n, k, loops, totals) for n in NAMES}

    p("\n" + "=" * 78)
    p(f" ROUND {k}/{loops} — 相对深度 {frac:.3f}")
    p("  层索引: " + " / ".join(f"{n.split('-')[-1]}={layers[n]}" for n in NAMES))
    p("=" * 78)

    # ---------- 数据采集 ----------
    pc = {}      # name -> (mc_signed, bl)  —— 凸包/面积必须用带符号值 (与 manifold_area.csv 同义)
    mca = {}     # name -> |Max_Corr|        —— 仅用于 Max_Corr 的分布检验
    ids = {}     # name -> dict lang->ID
    areas = {}   # name -> (area, norm)
    for n in NAMES:
        mc, bl = load_point_cloud(n, layers[n])
        pc[n] = (mc, bl)
        mca[n] = np.abs(mc) if len(mc) else mc
        ids[n] = load_id(n, layers[n])
        areas[n] = area_tbl.get((n, layers[n]), (None, None))

    # ID 逐语言对齐（取四模型共有语言，保证可比）
    common = set(lang_floor)
    for n in NAMES:
        common &= set(ids[n].keys())
    common = sorted(common)
    id_vec = {n: np.array([ids[n][l] for l in common]) for n in NAMES}

    # ---------- 聚合指标（描述性） ----------
    p("\n  聚合指标 (单值/模型层 — 描述性):")
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

    # ---------- 成对检验 ----------
    for pair, kind in [(LANG_PAIR, "同参数/不同语言数 (非缩放)"),
                       (SCALING_PAIR, "同语言数/不同参数 (真缩放)")]:
        a, b = pair
        p(f"\n  成对 {a} vs {b}  [{kind}]  (层 {layers[a]}/{layers[b]}):")
        p(effect_report((mca[a], mca[b]), "Max_Corr"))
        p(effect_report((pc[a][1], pc[b][1]), "Best_Lag_Sec"))
        p(effect_report((id_vec[a], id_vec[b]), "Intrinsic_Dim"))

        # 面积: bootstrap 差值
        # 凸包面积为带符号 (Max_Corr, Best_Lag) 的几何量, 与 manifold_area.csv 定义一致
        pa_pts = np.column_stack([pc[a][0], pc[a][1]]) if len(pc[a][0]) else np.zeros((0, 2))
        pb_pts = np.column_stack([pc[b][0], pc[b][1]]) if len(pc[b][0]) else np.zeros((0, 2))
        d, lo, hi = bootstrap_area_diff(pa_pts, pb_pts, seed=k)
        if np.isfinite(d):
            sig = "YES" if (lo > 0 or hi < 0) else " no"
            p(f"    {'Area(diff)':<14s} ΔA={d:+.4f} 95%CI [{lo:+.4f},{hi:+.4f}] excl0={sig}")
        else:
            p(f"    {'Area(diff)':<14s} insufficient samples")

    # ---------- 综合四模型 ----------
    p("\n  综合四模型 (omnibus):")
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

    # ---------- 缩放律（参数量 vs 聚合指标，n=4，精确 df=2） ----------
    p("\n  缩放律 (Pearson, n=4, 精确 df=2; |r|>=0.95 才 p<0.05):")
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
    """同时写屏幕与日志文件。"""
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
                print(f"[ERROR] --test 只接受 auto | nonparam", file=sys.stderr)
                return 2
            i += 2
        elif args[i] == "--no-log":
            use_log = False; i += 1
        elif args[i] in ("--root",):
            # 显式禁用: 本脚本只分析 FLEURS
            print("[ERROR] 本脚本固定使用 FLEURS 目录 re-organized/, 不支持 --root。",
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
    p(" 循环深度统计检验 — FLEURS (24 languages) · 六类指标")
    p("   Max_Corr / Best_Lag_Sec / Intrinsic_Dim / Nodes / Area / NormArea")
    p(f" 数据根目录: {DATA_ROOT}")
    p(f" 轮次: {loops}   每轮相对深度 = k/{loops}")
    p(f" scipy: {'available' if sp_stats is not None else 'MISSING'}"
      f"   检验方法: {'forced nonparametric' if FORCE_NONPARAM else 'auto (Shapiro->Welch/MW)'}")
    p("=" * 78)
    if sp_stats is None and not FORCE_NONPARAM:
        p("\n[WARNING] 未安装 scipy: 正态性检验无法执行, 所有指标将强制使用非参数检验")
        p("          (Mann-Whitney + Cliff's Δ)。装有 scipy 的机器在 auto 模式下会对正态指标")
        p("          改用 Welch t + Cohen's d, 因此两边效应量不可直接比较。")
        p("          需要跨环境可复现请加 --test nonparam (两边都强制非参数)。")

    if not os.path.isdir(DATA_ROOT):
        p(f"\n[ERROR] 未找到 FLEURS 数据目录: {DATA_ROOT}")
        p(f"        请把 FLEURS 数据放在脚本同级的 {FLEURS_ROOT_NAME}/ 下, 结构为:")
        p(f"          {FLEURS_ROOT_NAME}/<model>/formant_dim_analysis_cache/"
          f"Formant_Lag_Analysis_Results_moreLayers.csv")
        p(f"          {FLEURS_ROOT_NAME}/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv")
        p(f"          {FLEURS_ROOT_NAME}/manifold_area.csv")
        _flush(out); return 1

    totals = model_total_layers()
    area_tbl = load_area_table()

    # ---------- 启动诊断 ----------
    p("\n[诊断] 数据完整性检查")
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
        p(f"\n[ERROR] 缺少 Formant CSV (该模型的所有指标将为空): {', '.join(x.split(':')[0] for x in no_pc)}")
    if no_id:
        p(f"\n[INFO] 以下模型尚无 ID_result, ID 检验将自动跳过: "
          f"{', '.join(x.split(':')[0] for x in no_id)}")
        p("       若 ID 仍在计算中, 这是预期行为; 仅 Max_Corr / Best_Lag / Nodes / Area 参与统计。")
        p(f"       待 ID 生成后放入 {DATA_ROOT}/<model>/ID_result/ID_Analysis_L<layer>_Fixed.csv 即可。")
    if not missing:
        p("  -> 四个模型的 Formant CSV 与 ID_result 均已找到")

    # 数据集身份核验（语言数）
    probe = load_point_cloud(NAMES[0], 0)
    langs_pc = set()
    pc_path = os.path.join(DATA_ROOT, _model_dir(NAMES[0]), "formant_dim_analysis_cache",
                           "Formant_Lag_Analysis_Results_moreLayers.csv")
    if os.path.exists(pc_path):
        with open(pc_path, newline="", encoding="utf-8") as f:
            langs_pc = {str(r.get("Language", "")).strip() for r in csv.DictReader(f)}
    langs_pc.discard("")
    p(f"\n[数据集] {detect_dataset(len(langs_pc))}  (distinct Language = {len(langs_pc)})")
    if len(langs_pc) == 0:
        p("  [WARNING] 读不到语言名, 请确认路径与文件格式")
    elif len(langs_pc) < 20:
        p(f"  [ERROR] 语言数 {len(langs_pc)} < 20, 这不是 FLEURS (24 语言) 数据。")
        p("          本脚本仅用于 FLEURS; 请检查 re-organized/ 下是否为 FLEURS 结果。")
        _flush(out); return 3

    id_probe = load_id(NAMES[0], 0)
    lang_floor = sorted(id_probe.keys())
    if lang_floor:
        p(f"[ID] 模型 {NAMES[0]} 第0层 *_fleurs 语言数 = {len(lang_floor)}")
    else:
        p("[ID] 未检测到 ID 数据 -> Intrinsic_Dim 相关检验全部跳过 (其余指标正常)")

    # ---------- 逐轮 ----------
    summary = []
    for k in range(1, loops + 1):
        summary.append(run_round(k, loops, totals, area_tbl, lang_floor, out))
        p("")
        _flush(out)

    # ---------- 跨轮汇总 ----------
    p("\n" + "=" * 78)
    p(" 跨轮汇总 — 聚合指标随相对深度的变化")
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
    p(" 说明")
    p("-" * 78)
    p(" * Max_Corr / Best_Lag: 逐节点 (|Max_Corr|>0.1) 检验, 可做显著性推断")
    p(" * Intrinsic_Dim: 逐语言 (四模型共有的 *_fleurs 语言) 检验, 可做显著性推断")
    p(" * Nodes: 每层显著维度数, 单值 -> 仅描述性")
    p(" * Area / NormArea: 每层凸包面积, 单值 -> bootstrap CI + 差值 bootstrap 检验")
    p(" * 缩放律: 参数量 vs 聚合指标, Pearson 精确 t (n=4 -> df=2, 临界|r|=0.95)")
    p(" * 真缩放对比 = XLS-R-1B vs XLS-R-2B (同128语言/不同参数);")
    p("   XLSR-53 vs XLS-R-300M (同300M/不同语言数53vs128) 属数据规模效应, 非缩放。")
    p("=" * 78)

    _flush(out)

    if use_log and isinstance(sys.stdout, _Tee):
        print(f"\n[日志] 已保存: {log_path}", file=sys.__stdout__)
        sys.stdout.close()
    return 0


def _flush(out):
    """把累积的行一次性写出（便于 tee 到文件）。"""
    if not out:
        return
    sys.stdout.write("\n".join(out) + "\n")
    out.clear()


if __name__ == "__main__":
    sys.exit(main())
