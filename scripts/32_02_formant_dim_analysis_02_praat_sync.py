import numpy as np
import pandas as pd
import re
import os
from scipy.signal import correlate
from tqdm import tqdm

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

# --- 1. Configuration and Parameters ---
MAX_LAG_SEC = 2.0  # Maximum lag analysis range (±2 seconds)
SAMPLE_RATE = 10.0 # Step size of 0.1s
MAX_LAG_FRAMES = int(MAX_LAG_SEC * SAMPLE_RATE)

configs = [
    {   "MODEL" : "XLS-R-2B", 
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-2b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1920,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-2b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-xls-r-2b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLS-R-1B", 
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-1b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1280,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-1b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-xls-r-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLSR-53",
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-large-xlsr-53/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-large-xlsr-53/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-large-xlsr-53/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLS-R-300m",
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-300m/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-300m/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-xls-r-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "MMS-1B",
        "BASE_DIR" : r"./re-organized_mls/mms-1b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1280,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/mms-1b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/mms-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "MMS-300M",
        "BASE_DIR" : r"./re-organized_mls/mms-300m/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/mms-300m/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/mms-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "mHuBERT-147",
        "BASE_DIR" : r"./re-organized_mls/mHuBERT-147/",
        "LAYERS" : list(range(13)),
        "DIM_SIZE" : 768,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/mHuBERT-147/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/mHuBERT-147/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
]


# Previously locked 9 dimension experts (based on your provided data)
# LOCKED_EXPERTS = {
    # 8: [234, 1070, 728],
    # 25: [290, 1703, 525],
    # 38: [267, 163, 1844]
# }

# Language ISO mapping (same as before)
folder_to_iso = {
    'dutch': 'nld', 'english': 'eng',
    'french': 'fra', 'german': 'deu',
    'italian': 'ita', 'polish': 'pol',
    'portuguese': 'por', 'spanish': 'spa',
}

# --- 2. Core Parsing Function (Regex Extraction) ---
def parse_formant_grid(file_path):
    """
    Efficiently parses FormantGrid text and handles inconsistent lengths.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split F1, F2, F3...
    formant_sections = re.split(r'formants \[\d+\]:', content)[1:]
    
    raw_series = []
    max_len = 0
    
    for section in formant_sections[:3]: # Only take the first three formants
        values = re.findall(r'value = ([\d\.]+)', section)
        v_array = np.array([float(v) for v in values])
        raw_series.append(v_array)
        if len(v_array) > max_len:
            max_len = len(v_array)
            
    # --- Core Correction: Force Length Alignment ---
    # We base it on 36000 or the detected actual length, truncating or padding uniformly.
    # Here we adopt the "weakest link" principle, taking the minimum of the three to ensure a neat matrix.
    final_len = min([len(s) for s in raw_series])
    
    aligned_series = [s[:final_len] for s in raw_series]
    
    return np.array(aligned_series) # Shape is now necessarily (3, final_len)

# --- 3. Lag Cross-Correlation Calculation Logic ---
def get_best_lag_correlation(signal_a, signal_b):
    """
    Calculate the best lag time and corresponding maximum correlation coefficient for two sequences.
    """
    # Mean removal and normalization (Standardize)
    sig_a = (signal_a - np.mean(signal_a)) / (np.std(signal_a) + 1e-9)
    sig_b = (signal_b - np.mean(signal_b)) / (np.std(signal_b) + 1e-9)
    
    # Cross-correlation
    corr = correlate(sig_a, sig_b, mode='full') / len(sig_a)
    lags = np.arange(-len(sig_a) + 1, len(sig_a))
    
    # Restrict search range (MAX_LAG_FRAMES)
    mask = (lags >= -MAX_LAG_FRAMES) & (lags <= MAX_LAG_FRAMES)
    search_corr = corr[mask]
    search_lags = lags[mask]
    
    best_idx = np.argmax(np.abs(search_corr))
    return search_corr[best_idx], search_lags[best_idx] / SAMPLE_RATE


# --- 4. Main Program: Full Language Scan and Alignment ---
for config in configs:

    # --- Dynamically load locked expert dimensions ---
    # Please ensure the path matches the CACHE_DIR in the detection script
    expert_cache_path = config["CACHE_DIR"] + "/locked_timbre_experts_moreLayers.npy"

    if os.path.exists(expert_cache_path):
        # allow_pickle=True is required to read dictionary structures
        LOCKED_EXPERTS = np.load(expert_cache_path, allow_pickle=True).item()
        print(f"[*] Successfully loaded expert dimensions from cache: {LOCKED_EXPERTS}")
    else:
        print(f"[!] Error: Cache file {expert_cache_path} not found. Please run the detection script first.")
        # Provide a default empty value to prevent script crash
        LOCKED_EXPERTS = {}
        

    results = []

    for folder_prefix, iso in folder_to_iso.items():
        print(f"\nAligning language: {folder_prefix} (ISO: {iso})")
        
        # A. Extract physical formant data (Physical Reference)
        grid_path = os.path.join(r"./", "praat_PitchTier_mls", f"{folder_prefix}_1_hours_merged.FormantGrid")
        if not os.path.exists(grid_path):
            print(f"  [!] Missing FormantGrid: {grid_path}")
            continue
        
        formants = parse_formant_grid(grid_path) # (3, 36000)
        
        # B. Analyze locked dimensions
        for ly, dims in LOCKED_EXPERTS.items():
            # Load corresponding stable frame data
            npy_path = os.path.join(
                config["BASE_DIR"], f"hidden_states_{ly:02d}", 
                f"{folder_prefix}_mls", 
                f"hiddenlayer={ly}", 
                "X_flat_stable_1hours.npy"
            )
            
            if not os.path.exists(npy_path): continue
            
            # Load layer data (35998, 1920)
            data = np.load(npy_path, mmap_mode='r')
            
            for dim in dims:
                dim_signal = data[:, dim] # Neuron activation sequence
                
                # Align length (take minimum to prevent mismatch between 35998 vs 36000)
                n_min = min(len(dim_signal), formants.shape[1])
                s_neural = dim_signal[:n_min]
                
                # Calculate lag correlation for F1, F2, F3 respectively
                for f_idx in range(3):
                    s_formant = formants[f_idx, :n_min]
                    max_corr, best_lag = get_best_lag_correlation(s_neural, s_formant)
                    
                    results.append({
                        'Language': folder_prefix,
                        'Layer': ly,
                        'Dimension': dim,
                        'Formant': f"F{f_idx+1}",
                        'Max_Corr': max_corr,
                        'Best_Lag_Sec': best_lag
                    })

    # --- 5. Result Export and Summary Analysis ---
    df_final = pd.DataFrame(results)
    df_final.to_csv(os.path.join(config["CACHE_DIR"], "Formant_Lag_Analysis_Results_moreLayers.csv"), index=False)

    # Print qualitative examples of dimension functions (based on cross-language averages)
    print("\n" + "="*50)
    print("Qualitative Physical Function of Dimensions (Based on Cross-Language Averages)")
    summary = df_final.groupby(['Layer', 'Dimension', 'Formant'])[['Max_Corr', 'Best_Lag_Sec']].mean().reset_index()
    print(summary.sort_values(by=['Layer', 'Max_Corr'], ascending=[True, False]).head(15))

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()