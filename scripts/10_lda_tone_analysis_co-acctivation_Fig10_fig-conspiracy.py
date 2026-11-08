import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


# --- Basic Configuration (Please modify according to your paths) ---
BEST_LAYER = 38
TARGET_DIMS = [1815, 1680, 745, 386, 1765]  # Top 5 dimensions with the highest contribution in the plot
START_FRAME = 1000  # 20s
END_FRAME = 2000    # 40s
LANG_FOLDER = "_hn_rucheng"
BASE_DATA_DIR = configs[0]["BASE_DIR"] # Your data root directory

# Automatically construct paths
sub_dir = f"hiddenlayer={BEST_LAYER}"
file_path = os.path.join(BASE_DATA_DIR, f"hidden_states_{BEST_LAYER:02d}", LANG_FOLDER, sub_dir, "X_flat_stable_1hours.npy")

def analyze_top_dimensions_conspiracy():
    if not os.path.exists(file_path):
        print(f"Original file not found: {file_path}")
        return

    print(f">>> Reading large file via mmap and extracting Top dimension data...")
    # 1. Use mmap_mode to read; processing is extremely fast in a 128GB memory environment
    data = np.load(file_path, mmap_mode='r')
    
    # 2. Extract activation data for all specified Top dimensions within the time segment
    # The resulting shape will be (1000, 5)
    segments = data[START_FRAME:END_FRAME, TARGET_DIMS]
    
    # 3. Data preprocessing: Normalization (for easier subsequent plotting comparison)
    # Perform Min-Max normalization for each column separately
    segments_norm = (segments - np.min(segments, axis=0)) / (np.max(segments, axis=0) - np.min(segments, axis=0))

    # 4. Calculate Pearson correlation coefficient matrix
    # rowvar=False indicates that each column is a variable
    corr_matrix = np.corrcoef(segments, rowvar=False)

    # --- 5. Visualization section ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot A: Comparison of activation flows for Top 5 dimensions
    for i, dim in enumerate(TARGET_DIMS):
        ax1.plot(segments_norm[:, i], label=f"Dim {dim}", alpha=0.8, linewidth=1.2)
    ax1.set_title(f"Activation Flow Comparison (Layer {BEST_LAYER}, 20s-40s)", fontsize=14)
    ax1.set_ylabel("Normalized Activation")
    ax1.legend(loc='upper right', ncol=5)
    ax1.grid(True, alpha=0.2)

    # Plot B: Correlation heatmap (Conspiracy Matrix)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        # Add this line to control the font size within cells
        annot_kws={"size": 12}, 
        cmap='coolwarm', 
        center=0, 
        xticklabels=TARGET_DIMS, 
        yticklabels=TARGET_DIMS, 
        ax=ax2
    )

    ax2.set_title("Inter-Dimension Correlation Matrix (The Conspiracy Matrix)", fontsize=14)

    plt.tight_layout()
    plt.savefig("top_dimensions_conspiracy_analysis.png", dpi=300)
    print(">>> Analysis chart saved as: top_dimensions_conspiracy_analysis.png")
    plt.show()

    # 6. Save the extracted data segment for future use
    save_name = f"top_5_dims_segment_{START_FRAME}_{END_FRAME}.npy"
    np.save(save_name, segments)
    print(f">>> Extracted data segment saved to: {save_name}")

if __name__ == "__main__":
    analyze_top_dimensions_conspiracy()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()