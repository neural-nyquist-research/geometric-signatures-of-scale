import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

# --- 1. Core Configuration ---
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


# User-provided ISO mapping dictionary
folder_to_iso = {
    'dutch': 'nld', 'english': 'eng',
    'french': 'fra', 'german': 'deu',
    'italian': 'ita', 'polish': 'pol',
    'portuguese': 'por', 'spanish': 'spa',
}

# --- 2. Mean extraction function with caching mechanism ---
def get_layer_means_with_cache(BASE_DIR, CACHE_DIR, DIM_SIZE, layer):
    cache_path = os.path.join(CACHE_DIR, f"L{layer:02d}_mean_activations.csv")
    
    if os.path.exists(cache_path):
        print(f"[*] Layer {layer} cache found, loading directly...")
        return pd.read_csv(cache_path)
    
    print(f"[*] Cache not found, processing {DIM_SIZE}-dimensional data for Layer {layer} (35998 frames/language)...")
    layer_root = os.path.join(BASE_DIR, f"hidden_states_{layer:02d}")
    all_means = []

    # Iterate through all languages in the dictionary
    for folder_prefix, iso_code in tqdm(folder_to_iso.items(), desc=f"L{layer} Processing"):
        # Match folder name, e.g., 'arabic_mls'
        folder_name = f"{folder_prefix}_mls"
        file_path = os.path.join(
            layer_root, folder_name, 
            f"hiddenlayer={layer}", 
            "X_flat_stable_1hours.npy"
        )
        
        if os.path.exists(file_path):
            # Use mmap mode to prevent memory overflow
            data = np.load(file_path, mmap_mode='r')
            # Calculate mean activation over 35998 frames
            mean_vals = np.mean(data, axis=0)
            
            row = {"ISO639P3code": iso_code, "Folder": folder_name}
            for i, val in enumerate(mean_vals):
                row[f"D{i}"] = val
            all_means.append(row)
        else:
            print(f"[!] Missing data: {file_path}")

    df_means = pd.DataFrame(all_means)
    df_means.to_csv(cache_path, index=False)
    return df_means

# --- 3. Main program logic ---
for config in configs:
    print(f"Processing model: {config["MODEL"]}")
    if not os.path.exists(config["CACHE_DIR"]):
        os.makedirs(config["CACHE_DIR"])

    print("Loading typology data...")
    df_typo = pd.read_csv(config["TYPOLOGY_PATH"])

    final_candidates = []

    for ly in config["LAYERS"]:
        # A. Retrieve mean data (including caching logic)
        df_means = get_layer_means_with_cache(config["BASE_DIR"], config["CACHE_DIR"], config["DIM_SIZE"], ly)
        
        # B. Align based on ISO639P3code
        df_merged = pd.merge(df_means, df_typo, on='ISO639P3code')
        
        if df_merged.empty:
            print(f"[!] Layer {ly} alignment resulted in empty data; please check if ISO codes match.")
            continue

        # C. Cross-dimensional statistical scanning
        print(f"[*] Scanning typological sensitivity for Layer {ly}...")
        for d in tqdm(range(config["DIM_SIZE"]), desc=f"Stats Scanning L{ly}", leave=False):
            dim_col = f"D{d}"
            
            # Calculate correlation with Vowel Quality Inventories (Spearman)
            corr_v, p_v = spearmanr(df_merged[dim_col], df_merged['Vowel Quality Inventories'])
            # Calculate correlation with Consonant Inventories (reference metric)
            corr_c, p_c = spearmanr(df_merged[dim_col], df_merged['Consonant Inventories'])
            
            if p_v < 0.05: # Only record dimensions with significant correlation
                final_candidates.append({
                    'Layer': ly,
                    'Dimension': d,
                    'Vowel_Corr': corr_v,
                    'Vowel_P': p_v,
                    'Cons_Corr': corr_c
                })

    # --- 4. Export Final Report ---
    df_report = pd.DataFrame(final_candidates)
    df_report.to_csv(os.path.join(config["CACHE_DIR"], "Timbre_Expert_Final_Candidates_moreLayers.csv"), index=False)

    # --- 5. Automatically extract Top dimensions and save as npy cache ---
    locked_experts_cache = {}

    for ly in config["LAYERS"]:
        # Filter significant dimensions for the current layer, sorted by correlation absolute value descending
        layer_top = df_report[df_report['Layer'] == ly].copy()
        if not layer_top.empty:
            # Use abs() to sort considering both positive and negative correlation experts
            layer_top['abs_corr'] = layer_top['Vowel_Corr'].abs()
            top_dims = layer_top.sort_values(by='abs_corr', ascending=False).head(3)
            locked_experts_cache[ly] = top_dims['Dimension'].astype(int).tolist()
        else:
            locked_experts_cache[ly] = []

    # Save to config["CACHE_DIR"]
    npy_cache_path = os.path.join(config["CACHE_DIR"], "locked_timbre_experts_moreLayers.npy")
    np.save(npy_cache_path, locked_experts_cache)

    print(f"\n[OK] Top dimension expert list saved to: {npy_cache_path}")

    # 1. First report Top 3 by layer
    print("\n" + "="*50)
    print("Detection complete! Top 3 timbre candidate dimensions for each layer are as follows:")
    for ly in config["LAYERS"]:
        top = df_report[df_report['Layer'] == ly].sort_values(by='Vowel_Corr', ascending=False).head(3)
        print(f"\nLayer {ly}:")
        if top.empty:
            print("  No significantly correlated dimensions found.")
        else:
            for _, r in top.iterrows():
                print(f"  - Dim {int(r['Dimension'])}: Vowel_Corr={r['Vowel_Corr']:.3f} (P={r['Vowel_P']:.4e})")
    print("="*50)

    print("="*50)

    # 2. Exit loop: View cross-layer evolution of core dimension Dim 290 separately (key correction point)
    print("\n[Special Analysis] Correlation evolution of core dimension Dim 290 across layers:")
    # Ensure Dim 290 exists in df_report (i.e., it is significant)
    dim_290_data = df_report[df_report['Dimension'] == 290].sort_values(by='Layer')

    if not dim_290_data.empty:
        # Print Layer, Correlation Coefficient, and P-value
        print(dim_290_data[['Layer', 'Vowel_Corr', 'Vowel_P']].to_string(index=False))
    else:
        print("Notice: Dim 290 did not reach significance level (P<0.05) in any tested layer.")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()