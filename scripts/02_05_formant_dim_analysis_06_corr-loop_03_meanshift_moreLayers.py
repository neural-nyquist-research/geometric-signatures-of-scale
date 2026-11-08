import pandas as pd
import numpy as np

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
    {   "MODEL" : "MMS-1B",
        "BASE_DIR" : r"./re-organized/mms-1b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1280,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/mms-1b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/mms-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "MMS-300M",
        "BASE_DIR" : r"./re-organized/mms-300m/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/mms-300m/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/mms-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "mHuBERT-147",
        "BASE_DIR" : r"./re-organized/mHuBERT-147/",
        "LAYERS" : list(range(13)),
        "DIM_SIZE" : 768,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized/mHuBERT-147/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized/mHuBERT-147/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
]

def calculate_exact_mean_shift(csv_path, target_layer=38):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        return

    # 1. Load original correlation data
    df = pd.read_csv(csv_path)
    
    # 2. Filter dimensions for the specified layer with statistical significance
    # We use the common threshold |Max_Corr| > 0.1 seen in your logs
    layer_df = df[(df['Layer'] == target_layer) & (df['Max_Corr'].abs() > 0.1)].copy()
    
    if layer_df.empty:
        print(f"Warning: No significant dimensions found for Layer {target_layer}.")
        return

    # 3. Calculate the physical centroid for each language
    # Coordinate system is (Max_Corr, Best_Lag_Sec)
    centroids = layer_df.groupby('Language')[['Max_Corr', 'Best_Lag_Sec']].mean()
    
    # 4. Calculate the global centroid (average of all 24 languages' centroids)
    global_centroid = centroids.mean()
    
    # 5. Calculate the distance from each language's centroid to the global centroid (Shift)
    # We primarily focus on the offset along the time axis (Best_Lag_Sec), which is the source of the 41.2ms value
    # Alternatively, use Euclidean Distance
    shifts = np.sqrt(((centroids - global_centroid)**2).sum(axis=1))
    
    # 6. Calculate the average offset (Mean Shift)
    final_mean_shift = shifts.mean()
    
    print(f"\n" + "="*50)
    print(f"Model Layer: Layer {target_layer} | Total Languages: {len(centroids)}")
    print(f"Global Centroid Coordinates: Max_Corr={global_centroid['Max_Corr']:.4f}, Lag={global_centroid['Best_Lag_Sec']:.4f}s")
    print("-" * 50)
    
    # Print specific offset details for the top 5 languages as verification
    print("Partial language offset details (Sample):")
    for lang, shift in shifts.head(5).items():
        print(f"  - {lang}: {shift*1000:.2f} ms")
        
    print("-" * 50)
    print(f"Final Average Offset (Mean Shift): {final_mean_shift*1000:.2f} ms")
    print("="*50 + "\n")

for config in configs:
    print(f"Processing: {config['MODEL']}")
    # --- Please modify this section according to your actual file paths ---
    # Hint: Based on your previous logs, results may be saved in a file similar to formant_lag_analysis_results.csv
    results_file = config["csv_path"]

    for layer in config["LAYERS"]:
        calculate_exact_mean_shift(results_file, target_layer=layer)

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()