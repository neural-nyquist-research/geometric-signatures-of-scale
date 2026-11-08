import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# --- 1. Configuration and Paths ---
BEST_LAYER = 38
BASE_DATA_DIR = configs[0]["BASE_DIR"]
TYPOLOGY_CSV = configs[0]["TYPOLOGY_PATH"]  # Must include ISO639P3code and Tone (0 or 1) fields

# Language folder prefix to ISO code mapping
folder_to_iso = {
    "mandarin": "cmn", 
    "cantonese": "yue", 
    "vietnamese": "vie", 
    "thai": "tha",
    "english": "eng", 
    "spanish": "spa", 
    "italian": "ita", 
    "french": "fra",
    "russian": "rus", 
    "turkish": "tur", 
    "japanese": "jpn", 
    "korean": "kor",
    "arabic": "arb",
    "dutch": "nld",
    "portuguese": "por",
    "tamil": "tam",
    "zulu": "zul",
    # the seven FLEURS languages the earlier list omitted: without them the
    # projection covers 18 of the 25 languages and the caption's "25-language
    # corpus" is false
    "estonian": "ekk",
    "finnish": "fin",
    "georgian": "kat",
    "german": "deu",
    "indonesian": "ind",
    "polish": "pol",
    "swahili": "swh",
    "_hn_rucheng": "hak"
}

def run_tone_lda_analysis():
    # 1. Load typological data
    df_typeo = pd.read_csv(TYPOLOGY_CSV)
    
    X = []
    y = []
    labels = []
    
    layer_dir = os.path.join(BASE_DATA_DIR, f"hidden_states_{BEST_LAYER:02d}")
    
    print(f">>> Extracting L{BEST_LAYER} prototype vectors and matching tone labels...")
    
    for folder, iso in folder_to_iso.items():
        sub_dir = f"hiddenlayer={BEST_LAYER}"
        if folder == "_hn_rucheng":
            file_path = os.path.join(layer_dir, f"{folder}", sub_dir, "X_flat_stable_1hours.npy")
        else:
            file_path = os.path.join(layer_dir, f"{folder}_fleurs", sub_dir, "X_flat_stable_1hours.npy")
        
        if os.path.exists(file_path):
            # Extract prototype vector (Mean Pooling + L2 Norm)
            data = np.load(file_path, mmap_mode='r') # Open in read-only memory-mapped mode to avoid loading entirely into RAM
            proto = np.mean(data, axis=0)
            proto = proto / np.linalg.norm(proto)
            
            # Get tone label (Binary classification: Tone+ is 1, Tone- is 0)
            tone_label = 1 if df_typeo[df_typeo['ISO639P3code'] == iso]['Tone'].iloc[0] > 0.5 else 0
            
            X.append(proto)
            y.append(tone_label)
            labels.append(folder)

    X = np.array(X)
    y = np.array(y)

    # 2. Perform LDA
    # LDA seeks a projection direction that maximizes between-class variance while minimizing within-class variance
    lda = LinearDiscriminantAnalysis()
    print(f"Input feature dimension: {X.shape[1]}")
    X_lda = lda.fit_transform(X, y)
    
    # Extract feature weights (LDA coefficients)
    # The absolute value of the weights reflects which dimensions among the 1920 are most critical for Tone classification
    weights = np.abs(lda.coef_[0])
    top_indices = np.argsort(weights)[-20:][::-1]  # Get the top 20 dimensions with the highest contribution
    
    # 3. Visualization: Ranking of dimension contributions
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(range(20)), y=weights[top_indices], palette="viridis")
    plt.xticks(ticks=list(range(20)), labels=top_indices, rotation=45)
    plt.title(f"Top 20 Dimensions Contributing to [Tone] Classification (Layer {BEST_LAYER})", fontsize=14)
    plt.xlabel("Neuron Index (Dimension)", fontsize=12)
    plt.ylabel("Absolute LDA Coefficient", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"lda_tone_analysis_fig_01.png")

    plt.show()

    # --- Dump the projection; the figure itself is drawn by
    #     paper_revise/fig/replot_fig6_from_csv.py (Fig. 5 typography, Okabe-Ito
    #     palette, de-overlapped labels). Previously this block drew a 10x2 inch
    #     strip with 25 labels at 45 degrees, which collide at print width. ---
    rows = []
    for k, label in enumerate(labels):
        rows.append({
            "language": "Rucheng" if label == "_hn_rucheng" else label.capitalize(),
            "iso": folder_to_iso.get(label, ""),
            "tone": int(y[k]),
            "lda1": float(np.ravel(X_lda)[k]),
        })
    _here = os.path.dirname(os.path.abspath(__file__))
    _figdir = os.path.join(os.path.dirname(_here), "..", "data")
    proj_csv = os.path.join(_figdir if os.path.isdir(_figdir) else _here,
                            "lda_projection_L38.csv")
    pd.DataFrame(rows).sort_values("lda1").to_csv(proj_csv, index=False)
    print(f">>> projection written to {proj_csv} ({len(rows)} languages)")
    print(">>> now run: python figures/replot_fig6.py")

    return top_indices, weights

# Execute analysis
top_dims, all_weights = run_tone_lda_analysis()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()