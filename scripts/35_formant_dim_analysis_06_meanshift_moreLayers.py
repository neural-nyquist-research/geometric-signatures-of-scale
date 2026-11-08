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

def calculate_exact_mean_shift(csv_path, target_layer=38):
    results_tblentry={}
    
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        return None, results_tblentry          # was: return
        
    # 1. Load original correlation data
    df = pd.read_csv(csv_path)
    
    # 2. Filter dimensions for the specified layer with statistical significance
    # We use the common threshold |Max_Corr| > 0.1 seen in your logs
    layer_df = df[(df['Layer'] == target_layer) & (df['Max_Corr'].abs() > 0.1)].copy()
    
    if layer_df.empty:
        print(f"Warning: No significant dimensions found for Layer {target_layer}.")
        return None, results_tblentry          # was: return

    # 3. Calculate the physical centroid for each language
    # Coordinate system is (Max_Corr, Best_Lag_Sec)
    centroids = layer_df.groupby('Language')[['Max_Corr', 'Best_Lag_Sec']].mean()
    
    # 4. Calculate the global centroid (average of all 24 languages' centroids)
    global_centroid = centroids.mean()
    
    # 5. Calculate the distance from each language's centroid to the global centroid (Shift)
    shifts = np.sqrt(((centroids - global_centroid)**2).sum(axis=1))
    
    # 6. Calculate the average offset (Mean Shift)
    final_mean_shift = shifts.mean()
    
    print(f"\n" + "="*50)
    print(f"Model Layer: Layer {target_layer} | Total Languages: {len(centroids)}")
    print(f"Global centroid coordinates: Max_Corr={global_centroid['Max_Corr']:.4f}, Lag={global_centroid['Best_Lag_Sec']:.4f}s")
    print("-" * 50)
    
    results_tblentry.update({
                            "Layer": target_layer,
                            "N of Langs": len(centroids),
                            "Global centroid": global_centroid['Max_Corr']*1000,
                            "Global lag": global_centroid['Best_Lag_Sec']*1000
                            })
    
    # Print specific offset details for the first 5 languages as verification
    print("Partial language offset details (Sample):")
    for lang, shift in shifts.items():
        results_tblentry.update({ lang: shift*1000})
        if lang in ['french', 'cantonese', 'finnish', 'german', 'georgian']:
            print(f"  - {lang}: {shift*1000:.2f} ms")
    
    print("-" * 50)
    print(f"Final average offset (Mean Shift): {final_mean_shift*1000:.2f} ms")
    print("="*50 + "\n")
    
    return final_mean_shift*1000, results_tblentry

results_csv=[]
for config in configs:
    table_csv=[]
    print(f"Processing {config['MODEL']}")
    results_file = config["csv_path"]

    for layer in config["LAYERS"]:
        mean_shift, tblentry = calculate_exact_mean_shift(results_file, target_layer=layer)
        results_csv.append({"Model": config['MODEL'],
                            "Layer": layer,
                            "Mean Shift": mean_shift,
        })
        table_csv.append(tblentry)
    
    pd.DataFrame(table_csv).to_csv(os.path.join(config["BASE_DIR"], "mean_shift_by_langs.csv"), index=False)

pd.DataFrame(results_csv).to_csv(r"./re-organized_mls/mean_shift.csv", index=False)

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()