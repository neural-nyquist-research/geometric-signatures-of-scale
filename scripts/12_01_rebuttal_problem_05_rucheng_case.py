import numpy as np
from ripser import ripser
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


def get_rucheng_path(layer):
    xx = f"{layer:02d}"
    yy = str(layer)
    return os.path.join(configs[0]["BASE_DIR"], f"hidden_states_{xx}/_hn_rucheng/hiddenlayer={yy}/X_flat_stable_1hours.npy")

def analyze_rucheng_topology_with_checkpoints(output_csv=os.path.join(configs[0]["BASE_DIR"], "Rucheng_TDA_Evolution_Checkpointed.csv"), n_samples=1200):
    np.random.seed(42)
    
    # 1. Load existing results if they exist to resume progress
    if os.path.exists(output_csv):
        df_results = pd.read_csv(output_csv)
        completed_layers = df_results['Layer'].tolist()
        print(f"Resuming from checkpoint. {len(completed_layers)} layers already processed.")
    else:
        df_results = pd.DataFrame()
        completed_layers = []

    # 2. Iterate through all 49 layers
    for l in range(49):
        if l in completed_layers:
            continue
            
        path = get_rucheng_path(l)
        if not os.path.exists(path):
            print(f"Skipping Layer {l}: Path not found.")
            continue
            
        print(f"--- Processing Layer {l} ---")
        try:
            data = np.load(path)
            
            # Subsampling (Fixed seed 42 applied globally above)
            if data.shape[0] > n_samples:
                idx = np.random.choice(data.shape[0], n_samples, replace=False)
                data_sub = data[idx]
            else:
                data_sub = data

            # TDA: Compute H1 (Loops)
            dgms = ripser(data_sub, maxdim=1)['dgms']
            h1 = dgms[1]
            
            row = {'Layer': l, 'Max_H1_Persistence': 0, 'Normalized_H1_Persistence': 0, 'Num_Loops': 0}
            
            if len(h1) > 0:
                lifespans = h1[:, 1] - h1[:, 0]
                row['Max_H1_Persistence'] = np.max(lifespans)
                row['Normalized_H1_Persistence'] = np.max(lifespans) / np.max(h1[:, 1])
                row['Num_Loops'] = len(h1)
                # Mean of top 3 most persistent loops (The 'Logic Logic')
                row['Committee_H1_Mean'] = np.mean(np.sort(lifespans)[-3:])
            
            # 3. Append and Save immediately (Checkpointing)
            df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
            df_results.sort_values('Layer', inplace=True)
            df_results.to_csv(output_csv, index=False)
            print(f"Layer {l} saved to {output_csv}. Max Persistence: {row['Max_H1_Persistence']:.4f}")

        except Exception as e:
            print(f"Error at Layer {l}: {e}")
            continue

    return df_results

# Execute the pipeline
rucheng_tda_results = analyze_rucheng_topology_with_checkpoints()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()