import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

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


# Language folder prefix to ISO code mapping
folder_to_iso = {
    "english": "eng", 
    "spanish": "spa", 
    "italian": "ita", 
    "french": "fra",
    "dutch": "nld",      # Added
    "portuguese": "por", # Added
}

# ==================================================
# Starting automatic exploration and alignment analysis for Layer 8
# ==================================================
# Exploration complete! Tone-sensitive dimensions for Layer 8 are: [1210 1917  618]

# ==================================================
# Starting automatic exploration and alignment analysis for Layer 25
# ==================================================
# Exploration complete! Tone-sensitive dimensions for Layer 25 are: [ 582 1862  271]

# ==================================================
# Starting automatic exploration and alignment analysis for Layer 38
# ==================================================
# Exploration complete! Tone-sensitive dimensions for Layer 38 are: [1815 1680  745]



# folder_to_iso = {
    # 'arabic': 'arb', 'cantonese': 'yue', 'dutch': 'nld', 'english': 'eng',
    # 'estonian': 'ekk', 'finnish': 'fin', 'french': 'fra', 'georgian': 'kat',
    # 'german': 'deu', 'indonesian': 'ind', 'italian': 'ita', 'japanese': 'jpn',
    # 'korean': 'kor', 'mandarin': 'cmn', 'polish': 'pol', 'portuguese': 'por',
    # 'russian': 'rus', 'spanish': 'spa', 'swahili': 'swh', 'tamil': 'tam',
    # 'thai': 'tha', 'turkish': 'tur', 'vietnamese': 'vie', 'zulu': 'zul'
# }

# ==================================================
# Starting automatic exploration and alignment analysis for Layer 8
# ==================================================
# Exploration complete! Tone-sensitive dimensions for Layer 8 are: [ 773 1723 1232]

# ==================================================
# Starting automatic exploration and alignment analysis for Layer 25
# ==================================================
# Exploration complete! Tone-sensitive dimensions for Layer 25 are: [ 582 1383  271]

# ==================================================
# Starting automatic exploration and alignment analysis for Layer 38
# ==================================================
# Exploration complete! Tone-sensitive dimensions for Layer 38 are: [1876 1141  316]


# Suppress low-frequency warnings during computation
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. Global Configuration (Sync with your latest settings) ---
TOTAL_FRAMES = 36154
TOTAL_SECONDS = 3599  
STRIDE = TOTAL_SECONDS / TOTAL_FRAMES  # Approx 0.09997s
SEARCH_RANGE = 200                     # Search range suggestion: adjust based on STRIDE (200 frames ≈ 20s)

PITCH_TIER_DIR = "./praat_PitchTier_mls"
TYPOLOGY_CSV = "./cldf-datasets_wals/target_languages_typology.csv"  # Must include ISO639P3code and Tone (0 or 1) fields

def read_praat_pitchtier(file_path):
    """Read Praat PitchTier data"""
    if not os.path.exists(file_path): return None
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "number =" in lines[i] and i+1 < len(lines) and "value =" in lines[i+1]:
                t = float(lines[i].split('=')[-1].strip())
                v = float(lines[i+1].split('=')[-1].strip())
                data.append((t, v))
    return data

for config in configs:
    print(f"Processing {config["MODEL"]}")
    # Base path definition
    # Note: The script will automatically append the layer number after BASE_PATH, e.g., ./hidden_states_08
    BASE_PATH_PREFIX = config["BASE_DIR"].split("/")[-2] + "/hidden_states" 
    RESULT_DIR = f"{config["BASE_DIR"]}global_results"

    os.makedirs(RESULT_DIR, exist_ok=True)

    def get_top_dims_via_lda(layer_id, top_k=3):
        # 1. Load typological data
        df_typeo = pd.read_csv(TYPOLOGY_CSV)

        """
        Automatically explore the top k dimensions contributing most to tone classification in the specified layer
        """
        X, y = [], []
        layer_dir = f"re-organized_mls/{BASE_PATH_PREFIX}_{layer_id:02d}"
        
        # Iterate through folders to retrieve data (reuse your logic)
        for folder, iso in folder_to_iso.items():
            sub_dir = f"hiddenlayer={layer_id}"
            if folder == "_hn_rucheng":
                file_path = os.path.join(layer_dir, f"{folder}", sub_dir, "X_flat_stable_1hours.npy")
            else:
                file_path = os.path.join(layer_dir, f"{folder}_mls", sub_dir, "X_flat_stable_1hours.npy")


            if os.path.exists(file_path):
                data = np.load(file_path, mmap_mode='r')
                proto = np.mean(data, axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-9)
                
                # Get typological label (Tone+ is 1, Tone- is 0)
                tone_label = 1 if df_typeo[df_typeo['ISO639P3code'] == iso]['Tone'].iloc[0] > 0.5 else 0
                
                X.append(proto)
                y.append(tone_label)

        X, y = np.array(X), np.array(y)
        
        # Execute LDA to extract weights
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        
        # Find the top k indices with the largest absolute weights
        weights = np.abs(lda.coef_[0])
        top_indices = np.argsort(weights)[-top_k:][::-1]
        
        print(f"Exploration complete! Tone-sensitive dimensions for Layer {layer_id} are: {top_indices}")
        return top_indices.tolist()
        

    def analyze_layer_language(lang, layer_id, target_dims):
        # Dynamically construct the hierarchical path
        current_base_dir = f"re-organized_mls/{BASE_PATH_PREFIX}_{layer_id:02d}"
        
        if lang == "_hn_rucheng":
            npy_path = os.path.join(
                current_base_dir, 
                lang, 
                f"hiddenlayer={layer_id}", 
                "X_flat_stable_1hours.npy"
            )
            pt_path = os.path.join(PITCH_TIER_DIR, f"{lang[1:]}_1_hours_merged.PitchTier")
        else:
            npy_path = os.path.join(
                current_base_dir, 
                f"{lang}_mls", 
                f"hiddenlayer={layer_id}", 
                "X_flat_stable_1hours.npy"
            )
            pt_path = os.path.join(PITCH_TIER_DIR, f"{lang}_1_hours_merged.PitchTier")
        
        if not os.path.exists(npy_path):
            return None
        
        pitch_data = read_praat_pitchtier(pt_path)
        if pitch_data is None: return None

        # --- A. Data Alignment ---
        hidden_data = np.load(npy_path, mmap_mode='r')
        current_frames = hidden_data.shape[0]  # Detect actual frames (e.g., 35343)
        
        f0_aligned = np.full(current_frames, np.nan)
        
        for t, v in pitch_data:
            idx = int(round(t / STRIDE))
            if 0 <= idx < current_frames:
                f0_aligned[idx] = v
                
        f0_diff = np.abs(np.diff(f0_aligned, prepend=f0_aligned[0]))
        mask = ~np.isnan(f0_aligned) & ~np.isnan(f0_diff)
        
        lang_stats = []

        # --- B. Dimension Alignment Calculation ---
        for dim in target_dims:
            if dim >= hidden_data.shape[1]: continue
            
            neuron_act = np.array(hidden_data[:TOTAL_FRAMES, dim]) 
            clean_act = neuron_act[mask]
            clean_f0 = f0_aligned[mask]
            clean_diff = f0_diff[mask]
            
            if len(clean_f0) < 100: continue 
                
            # Basic correlation (Height & Velocity)
            corr_h = np.corrcoef(clean_act, clean_f0)[0, 1]
            corr_v = np.corrcoef(clean_act, clean_diff)[0, 1]
            
            # Lagged correlation search (standardization processing)
            n_act = (clean_act - np.mean(clean_act)) / (np.std(clean_act) + 1e-9)
            n_f0 = (clean_f0 - np.mean(clean_f0)) / (np.std(clean_f0) + 1e-9)
            
            lags = np.arange(-SEARCH_RANGE, SEARCH_RANGE)
            corrs = []
            target_act = n_act[SEARCH_RANGE:-SEARCH_RANGE]
            
            for l in lags:
                shifted_f0 = np.roll(n_f0, l)[SEARCH_RANGE:-SEARCH_RANGE]
                
                # Zero-bias check: prevent invalid calculations caused by constants
                if np.std(target_act) == 0 or np.std(shifted_f0) == 0:
                    c = 0.0
                else:
                    res = np.corrcoef(target_act, shifted_f0)[0, 1]
                    c = res if not np.isnan(res) else 0.0
                corrs.append(c)
                
            best_lag_f = lags[np.argmax(corrs)]
            best_lag_s = best_lag_f * STRIDE # Use your defined precise STRIDE
            
            lang_stats.append({
                "Layer": layer_id,
                "Language": lang,
                "Dimension": dim,
                "Corr_Height": corr_h,
                "Corr_Velocity": corr_v,
                "Best_Lag_Sec": best_lag_s,
                "Voice_Frames": len(clean_f0)
            })

        return lang_stats

    # --- 2. Batch Execution Logic ---
    # --- Experimental Configuration ---
    #TARGET_LAYERS = [8, 25, 38]
    TARGET_LAYERS = config["LAYERS"]
    #LANGUAGES = ["mandarin", "cantonese", "spanish", "vietnamese"]
    LANGUAGES = [
        "arabic", "cantonese", "dutch", "english", "estonian", "finnish", 
        "french", "georgian", "german", "indonesian", "italian", "japanese", 
        "korean", "mandarin", "polish", "portuguese", "russian", "spanish", 
        "swahili", "tamil", "thai", "turkish", "vietnamese", "zulu", "_hn_rucheng"
    ]
    FINAL_RESULTS = []

    for layer in TARGET_LAYERS:
        print(f"\n{'='*50}\nStarting automatic exploration and alignment analysis for Layer {layer}\n{'='*50}")
        
        # Step A: Dynamically discover the "key neurons" for this layer
        discovered_dims = get_top_dims_via_lda(layer, top_k=3)
        
        # Step B: Run cross-language Lag alignment for the discovered dimensions (reuse your previous analysis function)
        for lang in LANGUAGES:
            # analyze_layer_language is the integrated function from our previous round
            stats = analyze_layer_language(lang, layer, discovered_dims)
            if stats:
                FINAL_RESULTS.extend(stats)

    # Save the final cross-layer evolution report
    df_evolution = pd.DataFrame(FINAL_RESULTS)
    df_evolution.to_csv(f"{config["BASE_DIR"]}global_results/Layer_Evolution_Analysis_allLayers.csv", index=False)

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()