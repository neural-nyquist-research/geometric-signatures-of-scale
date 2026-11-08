import pandas as pd

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


def get_intrinsic_dim(model_type, layer_str):
    """
    Extract the Intrinsic_Dim of GLOBAL_MEAN from ID_Analysis_{LAYER}_Fixed.csv.
    """
    if model_type == 'XLSR':
        file_path = os.path.join(configs[2]["BASE_DIR"], f"ID_result/ID_Analysis_{layer_str}_Fixed.csv")
    elif model_type == 'XLSRTwoB':
        file_path = os.path.join(configs[0]["BASE_DIR"], f"ID_result/ID_Analysis_{layer_str}_Fixed.csv")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    mean_row = df[df['Language'] == 'GLOBAL_MEAN']
    
    if mean_row.empty:
        raise ValueError(f"GLOBAL_MEAN not found in {file_path}")
    
    return mean_row['Intrinsic_Dim'].values[0]

def get_entropy_values(model_name, layer_str):
    """
    Extract Entropy and H_norm from Normalized_Temporal_Entropy_Results.csv.
    """
    file_path = "./Normalized_Temporal_Entropy_Results.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    filtered_df = df[(df['Model'] == model_name) & (df['Layer'] == layer_str)]
    
    if filtered_df.empty:
        raise ValueError(f"Model {model_name}, Layer {layer_str} not found in {file_path}")
    
    row = filtered_df.iloc[0]
    return row['Entropy'], row['H_norm']

def format_val(val, decimals):
    """Format number, keeping specified decimal places"""
    return f"{val:.{decimals}f}"

def create_latex_row(model_key, layer, id_val, ent_val, norm_val, phase, 
                     bold_id=False, bold_ent=False, bold_norm=False, bold_phase=False):
    """
    Generate a single line of LaTeX code.
    model_key: 'XLSR' or 'XLSRTwoB'
    decimals_id: 2
    decimals_ent: 2
    decimals_norm: 3
    """
    # Format numbers
    # Intrinsic Dim: 2 decimals
    id_str = format_val(id_val, 2)
    # Temporal Entropy: 2 decimals (add ' bits')
    ent_str = f"{format_val(ent_val, 2)} bits"
    # Norm Entropy: 3 decimals
    norm_str = format_val(norm_val, 3)
    
    # Apply bold formatting
    id_latex = f"\\textbf{{{id_str}}}" if bold_id else id_str
    ent_latex = f"\\textbf{{{ent_str}}}" if bold_ent else ent_str
    norm_latex = f"\\textbf{{{norm_str}}}" if bold_norm else norm_str
    phase_latex = f"\\textbf{{{phase}}}" if bold_phase else phase
    
    # Model label
    model_latex = "\\XLSRFiveThree{{}}" if model_key == "XLSR" else "\\XLSRTwoB{{}}"
    
    return f"{model_latex} & {layer} & {id_latex} & {ent_latex} & {norm_latex} & {phase_latex}"

def main():
    print("Fetching data...")
    
    # 1. Fetch XLSR Data
    try:
        xlsr_l12_id = get_intrinsic_dim('XLSR', 'L12')
        xlsr_l12_ent, xlsr_l12_norm = get_entropy_values('XLSR-53', 'L12')
        
        xlsr_l13_id = get_intrinsic_dim('XLSR', 'L13')
        xlsr_l13_ent, xlsr_l13_norm = get_entropy_values('XLSR-53', 'L13')
    except Exception as e:
        print(f"Error fetching XLSR data: {e}")
        return

    # 2. Fetch XLSRTwoB Data
    try:
        xlsl2b_l23_id = get_intrinsic_dim('XLSRTwoB', 'L23')
        xlsl2b_l23_ent, xlsl2b_l23_norm = get_entropy_values('XLS-R-2B', 'L23')
        
        xlsl2b_l25_id = get_intrinsic_dim('XLSRTwoB', 'L25')
        xlsl2b_l25_ent, xlsl2b_l25_norm = get_entropy_values('XLS-R-2B', 'L25')
        
        xlsl2b_l27_id = get_intrinsic_dim('XLSRTwoB', 'L27')
        xlsl2b_l27_ent, xlsl2b_l27_norm = get_entropy_values('XLS-R-2B', 'L27')
        
        xlsl2b_l33_id = get_intrinsic_dim('XLSRTwoB', 'L33')
        xlsl2b_l33_ent, xlsl2b_l33_norm = get_entropy_values('XLS-R-2B', 'L33')
        
        xlsl2b_l35_id = get_intrinsic_dim('XLSRTwoB', 'L35')
        xlsl2b_l35_ent, xlsl2b_l35_norm = get_entropy_values('XLS-R-2B', 'L35')
    except Exception as e:
        print(f"Error fetching XLSRTwoB data: {e}")
        return

    print("Data fetched successfully. Generating LaTeX...")

    # Generate Rows based on the EXACT structure of the prompt's LaTeX code
    # Note: Boldings are inferred from the prompt's LaTeX code
    
    # Row 1: XLSR L12 (No bold) -> ends with \\ (no hline)
    row_l12 = create_latex_row("XLSR", "L12", xlsr_l12_id, xlsr_l12_ent, xlsr_l12_norm, "Pre-Bottleneck")
    
    # Row 2: XLSR L13 (All bold) -> ends with \\ \hline
    row_l13 = create_latex_row("XLSR", "L13", xlsr_l13_id, xlsr_l13_ent, xlsr_l13_norm, 
                               "Crystallization Peak (54\\%)", 
                               bold_id=True, bold_ent=True, bold_norm=True, bold_phase=True)
    
    # Row 3: XLSRTwoB L23 (No bold) -> ends with \\ (no hline)
    row_l23 = create_latex_row("XLSRTwoB", "L23", xlsl2b_l23_id, xlsl2b_l23_ent, xlsl2b_l23_norm, "Geometric Entry")
    
    # Row 4: XLSRTwoB L25 (No bold) -> ends with \\ (no hline)
    row_l25 = create_latex_row("XLSRTwoB", "L25", xlsl2b_l25_id, xlsl2b_l25_ent, xlsl2b_l25_norm, "Sustain Phase")
    
    # Row 5: XLSRTwoB L27 (Ent, Norm, Phase bold) -> ends with \\ (no hline)
    row_l27 = create_latex_row("XLSRTwoB", "L27", xlsl2b_l27_id, xlsl2b_l27_ent, xlsl2b_l27_norm, 
                               "Functional Peak (Bottleneck)", 
                               bold_ent=True, bold_norm=True, bold_phase=True)
    
    # Row 6: XLSRTwoB L33 (ID, Phase bold) -> ends with \\ (no hline)
    row_l33 = create_latex_row("XLSRTwoB", "L33", xlsl2b_l33_id, xlsl2b_l33_ent, xlsl2b_l33_norm, 
                               "Structural Peak", 
                               bold_id=True, bold_phase=True)
    
    # Row 7: XLSRTwoB L35 (Phase bold) -> ends with \\ \hline
    row_l35 = create_latex_row("XLSRTwoB", "L35", xlsl2b_l35_id, xlsl2b_l35_ent, xlsl2b_l35_norm, 
                               "Refinement Phase", 
                               bold_phase=True)

    # Construct the final LaTeX string
    latex_output = r"""
\begin{table}[ht]
\centering
\caption{Evolution of the Phonetic Bottleneck. The 2B model demonstrates a \textbf{Temporal Peak Shift}, utilizing a significantly wider functional window (up to L33/68\% depth) to support higher entropy through expanded dimensionality.}
\label{tab:id_ht_comparison}
\begin{adjustbox}{width=\columnwidth,center}
\begin{tabular}{lccccc}
\hline
\textbf{Model} & \textbf{Layer} & \textbf{Intrinsic Dim ($id$)} & \textbf{Temporal Entropy ($\Ht$)} & \textbf{Norm. Entropy ($\mathcal{H}_{\text{norm}}$)} & \textbf{Phase} \\ \hline
"""
    
    # Append rows with exact line endings as requested
    latex_output += f"{row_l12} \\\\\n"
    latex_output += f"{row_l13} \\\\ \hline\n"
    latex_output += f"{row_l23} \\\\\n"
    latex_output += f"{row_l25} \\\\\n"
    latex_output += f"{row_l27} \\\\\n"
    latex_output += f"{row_l33} \\\\\n"
    latex_output += f"{row_l35} \\\\ \hline\n"
    
    latex_output += r"""
\end{tabular}
\end{adjustbox}
\end{table}
"""

    print(latex_output)
    
    # Save to file
    with open("table_id_ht_comparison.tex", "w", encoding="utf-8") as f:
        f.write(latex_output)
    print("\nLaTeX code saved to 'table_id_ht_comparison.tex'")

if __name__ == "__main__":
    main()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()