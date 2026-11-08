import pandas as pd
import re
import math

from datetime import datetime
import os, sys

def _project_root():
    """The directory holding re-organized/. Current directory, then the script's, then its parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "re-organized")):
            return base
    return os.path.dirname(here)

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

configs = [
    {   "MODEL" : "XLS-R-2B", 
        "BASE_DIR" : r"./re-organized/wav2vec2-xls-r-2b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1920,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/wav2vec2-xls-r-2b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/wav2vec2-xls-r-2b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLS-R-1B", 
        "BASE_DIR" : r"./re-organized/wav2vec2-xls-r-1b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1280,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/wav2vec2-xls-r-1b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/wav2vec2-xls-r-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLSR-53",
        "BASE_DIR" : r"./re-organized/wav2vec2-large-xlsr-53/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/wav2vec2-large-xlsr-53/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/wav2vec2-large-xlsr-53/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLS-R-300m",
        "BASE_DIR" : r"./re-organized/wav2vec2-xls-r-300m/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/wav2vec2-xls-r-300m/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/wav2vec2-xls-r-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
]

def find_best_value(values, is_lower_better, is_physical):
    """
    Returns a list of LaTeX formatted strings.
    The best value is bolded.
    """
    if not values:
        return ["" for _ in values]
    
    # Find min or max depending on the metric
    best_val=[]
    if is_lower_better:
        best_val.append(min([float(v) for v in values[:3]]))
        best_val.append(min([float(v) for v in values[3:]]))
    else:
        best_val.append(max([float(v) for v in values[:3]]))
        best_val.append(max([float(v) for v in values[3:]]))
    print(best_val)
    latex_rows = []
    for x in range(len(values)):
        v=values[x]
        if float(v) == best_val[math.floor(x/3)]:
            if is_physical:
                v="("+str(v)+")"
            latex_rows.append(f"\\textbf{{{v}}}")
        else:
            if is_physical:
                v="("+str(v)+")"
            latex_rows.append(v)
    return latex_rows

def get_layer_value(df, layer_name):
    """
    Helper to get value for a specific layer.
    If the layer exists, return it. If not, return the last row's value as fallback.
    """
    if layer_name in df['Layer'].values:
        row = df[df['Layer'] == layer_name]
        return row.iloc[0]
    else:
        # Fallback to last layer if specific layer not found
        return df.iloc[-1]

def main():
    # Define file paths
    path_mean_shift = "./re-organized/mean_shift.csv"
    path_manifold_area = "./re-organized/manifold_area.csv"
    path_xl_sr_langs = os.path.join(configs[2]["BASE_DIR"],"mean_shift_by_langs.csv")
    path_xl_r_2b_langs = os.path.join(configs[0]["BASE_DIR"],"mean_shift_by_langs.csv")
    path_xl_r_1b_langs = os.path.join(configs[1]["BASE_DIR"],"mean_shift_by_langs.csv")
    path_xl_r_300_langs = os.path.join(configs[3]["BASE_DIR"],"mean_shift_by_langs.csv")

    # Load Global Data
    try:
        df_ms = pd.read_csv(path_mean_shift)
        df_ma = pd.read_csv(path_manifold_area)
    except FileNotFoundError as e:
        print(f"Error: Could not find global data files. {e}")
        return

    # Load Language Data
    try:
        df_xl_sr_langs = pd.read_csv(path_xl_sr_langs)
        df_xl_r_1b_langs = pd.read_csv(path_xl_r_1b_langs)
        df_xl_r_2b_langs = pd.read_csv(path_xl_r_2b_langs)
        df_xl_r_300_langs = pd.read_csv(path_xl_r_300_langs)
    except FileNotFoundError as e:
        print(f"Error: Could not find language data files. {e}")
        return

    # Define layers for each column
    layers = {
        "XLSR_PB": 13,       # Phonetic Bottleneck
        "XLSRThreeOO_PB": 13,       # Phonetic Bottleneck
        "XLSRTwoB_PB": 25,   # Phonetic Bottleneck
        "XLSR_DA": 19,       # Deep Alignment
        "XLSRThreeOO_DA": 19,       # Deep Alignment
        "XLSRTwoB_DA": 38    # Deep Alignment
    }

    # Helper to get MS value
    def get_ms_value(model_df, layer):
        row = get_layer_value(model_df, layer)
        return f"{row['Mean Shift']:.2f}"

    # Helper to get Area values
    def get_area_values(model_df, layer):
        row = get_layer_value(model_df, layer)
        norm_area = f"{row['Normalized Manifold Area']:.4f}"
        phys_area = f"{row['Physical Coverage Area']:.4f}"
        return norm_area, phys_area

    # --- Extract Global Stats ---
    
    # 1. Mean Shift (MS) [ms]
    # Values: XLSR(L13), XLSRTwoB(L25), XLSR(L19), XLSRTwoB(L38)
    ms_vals = [
        get_ms_value(df_ms[df_ms['Model'].str.contains('XLSR-53', case=False)], layers["XLSR_PB"]),
        get_ms_value(df_ms[df_ms['Model'].str.contains('XLS-R-300m', case=False)], layers["XLSRThreeOO_PB"]),
        get_ms_value(df_ms[df_ms['Model'].str.contains('XLS-R-2B', case=False)], layers["XLSRTwoB_PB"]),
        get_ms_value(df_ms[df_ms['Model'].str.contains('XLSR-53', case=False)], layers["XLSR_DA"]),
        get_ms_value(df_ms[df_ms['Model'].str.contains('XLS-R-300m', case=False)], layers["XLSRThreeOO_DA"]),
        get_ms_value(df_ms[df_ms['Model'].str.contains('XLS-R-2B', case=False)], layers["XLSRTwoB_DA"])
    ]
    print("!!!ms_vals")
    print(ms_vals)
    
    # 2. Norm. Area (A_hull)
    norm_area_vals = [
        get_area_values(df_ma[df_ma['Model'].str.contains('XLSR-53', case=False)], layers["XLSR_PB"])[0],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-300m', case=False)], layers["XLSRThreeOO_PB"])[0],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-2B', case=False)], layers["XLSRTwoB_PB"])[0],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLSR-53', case=False)], layers["XLSR_DA"])[0],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-300m', case=False)], layers["XLSRThreeOO_DA"])[0],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-2B', case=False)], layers["XLSRTwoB_DA"])[0]
    ]
    print("!!!norm_area_vals")
    print(norm_area_vals)
 
    # 3. Physical Area
    phys_area_vals = [
        get_area_values(df_ma[df_ma['Model'].str.contains('XLSR-53', case=False)], layers["XLSR_PB"])[1],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-300m', case=False)], layers["XLSRThreeOO_PB"])[1],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-2B', case=False)], layers["XLSRTwoB_PB"])[1],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLSR-53', case=False)], layers["XLSR_DA"])[1],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-300m', case=False)], layers["XLSRThreeOO_DA"])[1],
        get_area_values(df_ma[df_ma['Model'].str.contains('XLS-R-2B', case=False)], layers["XLSRTwoB_DA"])[1]
    ]
    print("!!!phys_area_vals")
    print(phys_area_vals)
    
    # Bold the best values for MS (lower is better) and Area (higher is better)
    ms_latex = find_best_value(ms_vals, is_lower_better=True, is_physical=False)
    norm_area_latex = find_best_value(norm_area_vals, is_lower_better=False, is_physical=False)
    phys_area_latex = find_best_value(phys_area_vals, is_lower_better=False, is_physical=True)

    # --- Extract Sample Languages Data ---
    # Languages of interest: French (fra), Cantonese (yue), Finnish (fin), German (deu), Georgian (kat)
    # Note: The CSV columns are lowercase language codes like 'french', 'cantonese', etc.
    
    lang_cols_map = {
        'French (fra)': 'french',
        'Cantonese (yue)': 'cantonese',
        'Finnish (fin)': 'finnish',
        'German (deu)': 'german',
        'Georgian (kat)': 'georgian'
    }

    sample_langs = list(lang_cols_map.keys())
    sample_latex_rows = []

    for lang_name in sample_langs:
        csv_col = lang_cols_map[lang_name]
        
        # Check if column exists
        if csv_col not in df_xl_sr_langs.columns or csv_col not in df_xl_r_2b_langs.columns:
            print(f"Warning: Column {csv_col} not found in language CSVs.")
            continue
            
        # Get values for each model/layer combo
        vals = []
        
        # XLSR PB (L13)
        row_xl_sr = get_layer_value(df_xl_sr_langs, layers["XLSR_PB"])
        if csv_col in row_xl_sr.index and pd.notna(row_xl_sr[csv_col]):
            vals.append(f"{row_xl_sr[csv_col]:.2f}")
        else:
            vals.append("-")
            
        # XLSRThreeOO PB (L13)
        row_xl_r_300 = get_layer_value(df_xl_r_300_langs, layers["XLSRThreeOO_PB"])
        if csv_col in row_xl_r_300.index and pd.notna(row_xl_r_300[csv_col]):
            vals.append(f"{row_xl_r_300[csv_col]:.2f}")
        else:
            vals.append("-")
            
        # XLSRTwoB PB (L25)
        row_xl_r_2b = get_layer_value(df_xl_r_2b_langs, layers["XLSRTwoB_PB"])
        if csv_col in row_xl_r_2b.index and pd.notna(row_xl_r_2b[csv_col]):
            vals.append(f"{row_xl_r_2b[csv_col]:.2f}")
        else:
            vals.append("-")
            
        # XLSR DA (L19)
        row_xl_sr_da = get_layer_value(df_xl_sr_langs, layers["XLSR_DA"])
        if csv_col in row_xl_sr_da.index and pd.notna(row_xl_sr_da[csv_col]):
            vals.append(f"{row_xl_sr_da[csv_col]:.2f}")
        else:
            vals.append("-")
            
        # XLSRThreeOO DA (L19)
        row_xl_r_300_da = get_layer_value(df_xl_r_300_langs, layers["XLSRThreeOO_DA"])
        if csv_col in row_xl_r_300_da.index and pd.notna(row_xl_r_300_da[csv_col]):
            vals.append(f"{row_xl_r_300_da[csv_col]:.2f}")
        else:
            vals.append("-")
            
        # XLSRTwoB DA (L38)
        row_xl_r_2b_da = get_layer_value(df_xl_r_2b_langs, layers["XLSRTwoB_DA"])
        if csv_col in row_xl_r_2b_da.index and pd.notna(row_xl_r_2b_da[csv_col]):
            vals.append(f"{row_xl_r_2b_da[csv_col]:.2f}")
        else:
            vals.append("-")
            
        # Bold best value for this row (lower is better for MS)
        # Filter out non-numeric values for comparison
        numeric_vals = [float(v) for v in vals if v != "-"]
        if numeric_vals:
            best_val=[]
            best_val.append(min(numeric_vals[:3]))
            best_val.append(min(numeric_vals[3:]))
            
            latex_vals = []
            for x in range(len(vals)):
                v = vals[x]
                if v != "-" and float(v) == best_val[math.floor(x/3)]:
                    latex_vals.append(f"\\textbf{{{v}}}")
                else:
                    latex_vals.append(v)
            sample_latex_rows.append(f"{lang_name} & {' & '.join(latex_vals)} \\\\")
        else:
            sample_latex_rows.append(f"{lang_name} & {' & '.join(vals)} \\\\")

    ms_latex_str = " & ".join(ms_latex)
    sample_rows_str = "\n".join(sample_latex_rows)
    norm_area_str = " & ".join(norm_area_latex)
    phys_area_str = " & ".join(phys_area_latex)
    # --- Construct LaTeX Table ---
    # Use an f-string for easier reading, but escape ALL LaTeX braces {{ and }}
    # The placeholders for data will be {ms_latex}, etc.
    
    # Note: In f-strings, {{ becomes { and }} becomes }
    # So \begin{{tabular}} becomes \begin{tabular} in the output.
    
    latex_table = f"""\\begin{{table}}[t]
\\centering
\\caption{{Cross-lingual Alignment Stability ($MS$) and Manifold Capacity ($A_{{\\text{{hull}}}}$).}}
\\label{{tab:alignment_comparison}}
\\scriptsize
\\begin{{tabular}}{{l@{{\\quad}}c@{{\\quad}}c@{{\\quad}}c@{{\\quad}}c@{{\\quad}}c@{{\\quad}}c}}
\\toprule
\\textbf{{Metric}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Phonetic Bottleneck}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Deep Alignment}}}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
& \\textbf{{53}} & \\textbf{{300}} & \\textbf{{2B}} & \\textbf{{53}} & \\textbf{{300}} & \\textbf{{2B}} \\\\
& \\textbf{{(L13)}} & \\textbf{{(L13)}} & \\textbf{{(L25)}} & \\textbf{{(L19)}} & \\textbf{{(L19)}} & \\textbf{{(L38)}} \\\\
\\midrule
\\textbf{{Global Stats}} & & & & \\\\
Mean Shift [ms] $\\downarrow$ & {ms_latex_str} \\\\
Norm. Area ($A_{{\\text{{hull}}}}$) $\\uparrow$ & {norm_area_str} \\\\
(Physical Area) & {phys_area_str} \\\\
\\midrule
\\textbf{{Samples ($MS$ [ms])}} & & & & \\\\
{sample_rows_str}
\\bottomrule
\\end{{tabular}}
\\begin{{flushleft}}
\\footnotesize
Note: The \\XLSRTwoB{{}} model utilizes its expanded bottleneck ($A_{{\\text{{hull}}}}=0.0875$) to sustain a ``Broad-Nebula'' manifold, avoiding the early saturation seen in smaller architectures like \\XLSRFiveThree{{}} ($A_{{\\text{{hull}}}}=0.0775$). This geometric clearance enables the 2B model to delay specialization and support outlier dialects (e.g., Rucheng) without sacrificing physical alignment. In the deep alignment stage, this manifests as the lowest Mean Shift (MS = 42.60 ms), reflecting robust phonetic grounding — though this grounding is language-specific rather than universal, as evidenced by late-stage variance expansion in Fig.~\\ref{{fig:grand_synthesis}}.
\\end{{flushleft}}
\\end{{table}}
"""

    print(latex_table)

    # Optional: Save to file
    _out = os.path.join(_project_root(), 'results')
    os.makedirs(_out, exist_ok=True)
    with open(os.path.join(_out, 'table_alignment_comparison.tex'), 'w') as f:
        f.write(latex_table)
    print("Table saved to alignment_comparison.tex")
    
if __name__ == "__main__":
    main()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()