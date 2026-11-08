import pandas as pd

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

def get_value_from_csv(filepath, model_name, layer_num, column_name):
    """
    Reads a CSV file and returns the value for a specific model and layer.
    If the specific layer is not found, it returns the maximum value in that column for that model.
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Filter by model
    if model_name == 'XLSR':
        model_filter = 'XLSR-53'
    elif model_name == 'XLSRTwoB':
        model_filter = 'XLS-R-2B'
    else:
        model_filter = model_name

    model_df = df[df['Model'] == model_filter]

    if model_df.empty:
        print(f"Warning: Model {model_filter} not found in {filepath}")
        return None

    # Check if 'Layer' column exists and is numeric
    if 'Layer' in model_df.columns:
        # Try to find the specific layer
        specific_layer_df = model_df[model_df['Layer'] == layer_num]
        
        if not specific_layer_df.empty:
            val = specific_layer_df[column_name].values[0]
            return val
        else:
            # Fallback: Use the maximum value in the column for this model
            print(f"Info: Layer {layer_num} not found in {filepath} for {model_filter}. Using max value.")
            val = model_df[column_name].max()
            return val
    else:
        # If no Layer column, just take the first or max
        print(f"Info: No Layer column in {filepath}. Using max value.")
        val = model_df[column_name].max()
        return val

def format_value(val, decimal_places=3):
    """Formats a float value to a string with specified decimal places."""
    if val is None:
        return "N/A"
    return f"{val:.{decimal_places}f}"

def calculate_change_pct(val1, val2):
    """
    Calculates percentage change from val1 to val2.
    Returns a formatted string like '+15.7%' or '-29.0%'
    """
    if val1 is None or val2 is None:
        return "N/A"
    if val1 == 0:
        return "N/A"
    change = ((val2 - val1) / abs(val1)) * 100
    if change >= 0:
        return f"+{change:.1f}\\%"
    else:
        return f"{change:.1f}\\%"

def main():
    # Define file paths
    entropy_csv = './re-organized/Normalized_Temporal_Entropy_Results.csv'
    manifold_csv = './re-organized/manifold_area.csv'
    mean_shift_csv = './re-organized/mean_shift.csv'

    # Define models and layers as per caption: XLSR L13, XLSRTwoB L25
    model_xlsr = 'XLSR'
    model_xlsrtwob = 'XLSRTwoB'
    layer_xlsr = 13
    layer_xlsrtwob = 25

    # 1) Normalized Temporal Entropy (H_norm)
    h_norm_xlsr = get_value_from_csv(entropy_csv, model_xlsr, layer_xlsr, 'H_norm')
    h_norm_xlsrtwob = get_value_from_csv(entropy_csv, model_xlsrtwob, layer_xlsrtwob, 'H_norm')

    # 2) Temporal Entropy (Entropy)
    entropy_xlsr = get_value_from_csv(entropy_csv, model_xlsr, layer_xlsr, 'Entropy')
    entropy_xlsrtwob = get_value_from_csv(entropy_csv, model_xlsrtwob, layer_xlsrtwob, 'Entropy')

    # 3) Normalized Manifold Area (Normalized Manifold Area)
    norm_manifold_xlsr = get_value_from_csv(manifold_csv, model_xlsr, layer_xlsr, 'Normalized Manifold Area')
    norm_manifold_xlsrtwob = get_value_from_csv(manifold_csv, model_xlsrtwob, layer_xlsrtwob, 'Normalized Manifold Area')

    # 4) Physical Coverage Area
    phys_coverage_xlsr = get_value_from_csv(manifold_csv, model_xlsr, layer_xlsr, 'Physical Coverage Area')
    phys_coverage_xlsrtwob = get_value_from_csv(manifold_csv, model_xlsrtwob, layer_xlsrtwob, 'Physical Coverage Area')

    # 5) Mean Shift (Mean Shift)
    mean_shift_xlsr = get_value_from_csv(mean_shift_csv, model_xlsr, layer_xlsr, 'Mean Shift')
    mean_shift_xlsrtwob = get_value_from_csv(mean_shift_csv, model_xlsrtwob, layer_xlsrtwob, 'Mean Shift')

    # Format values
    h_norm_xlsr_str = format_value(h_norm_xlsr, 3)
    h_norm_xlsrtwob_str = format_value(h_norm_xlsrtwob, 3)
    h_norm_change = calculate_change_pct(h_norm_xlsr, h_norm_xlsrtwob)

    entropy_xlsr_str = f"{format_value(entropy_xlsr, 2)} bits"
    entropy_xlsrtwob_str = f"{format_value(entropy_xlsrtwob, 2)} bits"
    entropy_change = calculate_change_pct(entropy_xlsr, entropy_xlsrtwob)

    norm_manifold_xlsr_str = format_value(norm_manifold_xlsr, 4)
    norm_manifold_xlsrtwob_str = format_value(norm_manifold_xlsrtwob, 4)
    norm_manifold_change = calculate_change_pct(norm_manifold_xlsr, norm_manifold_xlsrtwob)

    phys_coverage_xlsr_str = format_value(phys_coverage_xlsr, 4)
    phys_coverage_xlsrtwob_str = format_value(phys_coverage_xlsrtwob, 4)
    phys_coverage_change = calculate_change_pct(phys_coverage_xlsr, phys_coverage_xlsrtwob)

    mean_shift_xlsr_str = f"{format_value(mean_shift_xlsr, 2)} ms"
    mean_shift_xlsrtwob_str = f"{format_value(mean_shift_xlsrtwob, 2)} ms"
    mean_shift_change = calculate_change_pct(mean_shift_xlsr, mean_shift_xlsrtwob)

    # Bold the change if it's significant (e.g., > 20% or < -20%) as in the example
    def bold_if_significant(change_str):
        if change_str == "N/A":
            return change_str
        try:
            val = float(change_str.replace('+', '').replace('\\%', ''))
            if abs(val) > 20:  # Arbitrary threshold based on example where 44.7% and -29.0% are bold
                return f"\\textbf{{{change_str}}}"
        except:
            pass
        return change_str

    h_norm_change_b = bold_if_significant(h_norm_change)
    entropy_change_b = bold_if_significant(entropy_change)
    norm_manifold_change_b = bold_if_significant(norm_manifold_change)
    phys_coverage_change_b = bold_if_significant(phys_coverage_change)
    mean_shift_change_b = bold_if_significant(mean_shift_change)

    # Generate LaTeX code
    latex_table = f"""\\begin{{table}}[H]
\\centering
\\caption{{Comparative Metrics at Model Functional Peaks (\\XLSR{{}} L13 vs. \\XLSRTwoB{{}} L25)}}
\\label{{tab:scaling_metrics}}
\\begin{{tabularx}}{{\\columnwidth}}{{Xccr}}
\\hline
\\textbf{{Metric}} & \\textbf{{\\XLSRFiveThree{{}}}} & \\textbf{{\\XLSRTwoB{{}}}} & \\textbf{{Change ($\\Delta$)}} \\\\ \\hline
Normalized Temporal Entropy ($\\mathcal{{H}}_{{\\text{{norm}}}}$) & {h_norm_xlsr_str} & {h_norm_xlsrtwob_str} & {h_norm_change_b} \\\\
Temporal Entropy & {entropy_xlsr_str} & {entropy_xlsrtwob_str} & {entropy_change_b} \\\\
Normalized Manifold Area ($A_{{\\text{{hull}}}}$) & {norm_manifold_xlsr_str} & {norm_manifold_xlsrtwob_str} & {norm_manifold_change_b} \\\\
Physical Coverage Area & {phys_coverage_xlsr_str} & {phys_coverage_xlsrtwob_str} & {phys_coverage_change_b} \\\\
Mean Shift ($MS$) & {mean_shift_xlsr_str} & {mean_shift_xlsrtwob_str} & {mean_shift_change_b} \\\\ \\hline
\\end{{tabularx}}
\\end{{table}}"""

    print(latex_table)

    # Optional: Save to file
    _out = os.path.join(_project_root(), 'results')
    os.makedirs(_out, exist_ok=True)
    with open(os.path.join(_out, 'scaling_metrics.tex'), 'w') as f:
        f.write(latex_table)
    print("Table saved to scaling_metrics.tex")

if __name__ == "__main__":
    main()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()