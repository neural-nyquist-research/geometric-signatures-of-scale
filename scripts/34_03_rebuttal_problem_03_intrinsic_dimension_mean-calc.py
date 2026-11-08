import os
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

def append_mean_to_csvs(base_path, total_layers):
    """
    Computes the mean for each CSV file and appends it as a new row.
    """
    print(f"Processing directory: {base_path}")
    
    for layer in range(total_layers + 1):
        file_path = os.path.join(base_path, f"ID_Analysis_L{layer}_Fixed.csv")
        
        if os.path.exists(file_path):
            # Load the individual layer file
            df = pd.read_csv(file_path)
            
            # Select only numeric columns for the mean calculation
            # (Ensures we don't try to average language names or families)
            numeric_cols = df.select_dtypes(include=['number']).columns
            mean_values = df[numeric_cols].mean()
            
            # Create the summary row
            # We set a distinct identifier in the first column (usually 'Language' or 'Index')
            mean_row = pd.DataFrame([mean_values])
            mean_row.insert(0, df.columns[0], 'GLOBAL_MEAN')
            
            # Append and save back to the same path
            updated_df = pd.concat([df, mean_row], ignore_index=True)
            updated_df.to_csv(file_path, index=False)
            
            print(f"  - Updated Layer {layer}")
        else:
            print(f"  - Skip: Layer {layer} not found.")

# Define paths for your two architectures
path_2b = os.path.join(configs[0]["BASE_DIR"], "ID_result")
path_1b = os.path.join(configs[1]["BASE_DIR"], "ID_result")
path_53 = os.path.join(configs[2]["BASE_DIR"], "ID_result")
path_300 = os.path.join(configs[3]["BASE_DIR"], "ID_result")
path_mms1b = os.path.join(configs[4]["BASE_DIR"], "ID_result")
path_mms300 = os.path.join(configs[5]["BASE_DIR"], "ID_result")
path_mHuBERT = os.path.join(configs[6]["BASE_DIR"], "ID_result")

# Execute the updates
append_mean_to_csvs(path_2b, len(configs[0]["LAYERS"])-1)
append_mean_to_csvs(path_1b, len(configs[1]["LAYERS"])-1)
append_mean_to_csvs(path_53, len(configs[2]["LAYERS"])-1)
append_mean_to_csvs(path_300, len(configs[3]["LAYERS"])-1)
append_mean_to_csvs(path_mms1b, len(configs[4]["LAYERS"])-1)
append_mean_to_csvs(path_mms300, len(configs[5]["LAYERS"])-1)
append_mean_to_csvs(path_mHuBERT, len(configs[6]["LAYERS"])-1)

print("\nAll files updated with GLOBAL_MEAN rows.")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()