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
    "_hn_rucheng": "hak"
}

def run_tone_lda_analysis():
    # 1. Load typology data
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
            data = np.load(file_path, mmap_mode='r') # Open with read-only memory mapping to avoid loading entirely into RAM
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
    # The absolute value of the weight reflects which dimensions among 1920 are most critical for Tone classification
    weights = np.abs(lda.coef_[0])
    top_indices = np.argsort(weights)[-20:][::-1]  # Take the top 20 dimensions with highest contribution
    
    return top_indices, weights


def run_syllable_lda_analysis():
    # 1. Load typology data
    df_typeo = pd.read_csv(TYPOLOGY_CSV)
    
    X = []
    y = []
    labels = []
    
    layer_dir = os.path.join(BASE_DATA_DIR, f"hidden_states_{BEST_LAYER:02d}")
    
    print(f">>> Extracting L{BEST_LAYER} prototype vectors and matching syllable structure labels (3-class)...")
    
    for folder, iso in folder_to_iso.items():
        sub_dir = f"hiddenlayer={BEST_LAYER}"
        # Compatible with your path structure
        if folder == "_hn_rucheng":
            file_path = os.path.join(layer_dir, folder, sub_dir, "X_flat_stable_1hours.npy")
        else:
            file_path = os.path.join(layer_dir, f"{folder}_fleurs", sub_dir, "X_flat_stable_1hours.npy")
        
        if os.path.exists(file_path):
            data = np.load(file_path, mmap_mode='r')
            proto = np.mean(data, axis=0)
            proto = proto / np.linalg.norm(proto)
            
            # --- 3-class classification logic ---
            raw_val = df_typeo[df_typeo['ISO639P3code'] == iso]['Syllable Structure'].iloc[0]
            if raw_val > 0.7:
                syll_label = 2  # Complex (e.g., English, French, Polish)
            elif raw_val > 0.4:
                syll_label = 1  # Medium (e.g., Mandarin, Arabic, Cantonese)
            else:
                syll_label = 0  # Simple (e.g., Swahili/Zulu)
            
            X.append(proto)
            y.append(syll_label)
            labels.append(folder)

    X = np.array(X)
    y = np.array(y)

    # 2. Perform LDA
    # For three-class problems, n_components can be at most 2
    
    # --- Fixed LDA initialization ---
    # Get the actual number of existing classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # Dynamically set n_components to ensure it does not exceed (number of classes - 1)
    target_components = min(2, n_classes - 1)
    
    if target_components < 1:
        raise ValueError(f"Only {n_classes} class(es) found in current samples; LDA classification cannot be performed. Please check the CSV label distribution.")

    lda = LinearDiscriminantAnalysis(n_components=target_components)
    print(f"Input feature dimension: {X.shape[1]}, Actual number of classes: {n_classes}, Setting n_components: {target_components}")
    
    X_lda = lda.fit_transform(X, y)
    # -----------------------
    
    # Extract feature weights (take the square root of the sum of squares of coefficients for both discriminant axes, representing total contribution to classification)
    weights = np.sqrt(np.sum(np.square(lda.coef_), axis=0))
    top_indices = np.argsort(weights)[-20:][::-1]

    return top_indices, weights

def analyze_feature_decoupling(tone_weights, syll_weights, top_n=20):
    """
    Compare the overlap of feature dimensions between tone and syllable structure.
    tone_weights: Absolute LDA weights for 1920-dimensional tone features
    syll_weights: L2 Norm of LDA weights for 1920-dimensional syllable structure features
    """
    # 1. Extract indices of Top N dimensions
    top_tone_idx = set(np.argsort(tone_weights)[-top_n:][::-1])
    top_syll_idx = set(np.argsort(syll_weights)[-top_n:][::-1])
    
    # 2. Calculate intersection (overlapping neurons)
    intersection = top_tone_idx.intersection(top_syll_idx)
    union = top_tone_idx.union(top_syll_idx)
    jaccard_sim = len(intersection) / len(union)
    
    print(f"--- Dimension Decoupling Analysis (Top {top_n}) ---")
    print(f"Tone core dimensions: {sorted(list(top_tone_idx))}")
    print(f"Syllable core dimensions: {sorted(list(top_syll_idx))}")
    print(f"Number of overlapping dimensions: {len(intersection)}")
    if len(intersection) > 0:
        print(f"Overlapping dimension indices: {intersection}")
    else:
        print("Conclusion: Complete decoupling found! The two feature types are encoded by mutually non-overlapping neuron populations.")
    print(f"Jaccard similarity: {jaccard_sim:.4f}")

    # 3. Visualization: Scatter plot of weight correlations
    plt.figure(figsize=(8, 8))
    plt.scatter(tone_weights, syll_weights, alpha=0.3, s=10, c='purple')
    
    # Annotate Top dimensions
    for idx in list(top_tone_idx)[:10]:
        plt.annotate(f"Tone_{idx}", (tone_weights[idx], syll_weights[idx]), fontsize=8, color='blue')
    for idx in list(top_syll_idx)[:10]:
        plt.annotate(f"Syll_{idx}", (tone_weights[idx], syll_weights[idx]), fontsize=8, color='red')
        
    plt.title(f"Functional Disentanglement: Tone vs Syllable (Layer 38)", fontsize=14)
    plt.xlabel("Tone LDA Importance", fontsize=12)
    plt.ylabel("Syllable LDA Importance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Draw simplified Venn diagram logic for overlap (using text table instead of library dependency)
    plt.text(0.7, 0.7, f"Intersection: {len(intersection)}\nSimilarity: {jaccard_sim:.2%}", 
             transform=plt.gca().transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig("feature_disentanglement_analysis.png", bbox_inches='tight')
    plt.show()

# Execute analysis
top_dims_tone, all_weights_tone = run_tone_lda_analysis()
top_dims_syll, all_weights_syll = run_syllable_lda_analysis()

analyze_feature_decoupling(all_weights_tone, all_weights_syll)

# --- Dump the two per-dimension weight vectors so the figure can be re-rendered at
#     print size by figures/replot_fig7.py. The PNG this
#     script writes is 8x8 in at low dpi: square, so at a legible font size it would
#     take ~5-7 in of page height. The dump is the portable part. ---
_here = os.path.dirname(os.path.abspath(__file__))
_figdir = os.path.join(os.path.dirname(_here), "..", "data")
_out = os.path.join(_figdir if os.path.isdir(_figdir) else _here, "lda_weights_L38.csv")
with open(_out, "w", encoding="utf-8") as fh:
    fh.write("dimension,tone_weight,syllable_weight\n")
    for d in range(len(all_weights_tone)):
        fh.write("%d,%.10g,%.10g\n" % (d, float(all_weights_tone[d]), float(all_weights_syll[d])))
print(f">>> weights written to {_out} ({len(all_weights_tone)} dimensions)")
print(">>> now run: python figures/replot_fig7.py")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()