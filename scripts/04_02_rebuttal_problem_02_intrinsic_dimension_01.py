import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
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

# Set seeds for reproducibility
np.random.seed(42)

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

def estimate_intrinsic_dimension(X):
    # 1. Cast and Filter
    X = X.astype(np.float32)
    norms = np.linalg.norm(X, axis=1)
    threshold = np.percentile(norms, 20) 
    X_active = X[norms > threshold]
    
    # 2. UNIT SPHERE NORMALIZATION
    # In high-dim spaces (1920D), distances are more stable on the unit sphere
    # This prevents 'Global Max 247' from blowing up the distance variance
    X_active = X_active / (np.linalg.norm(X_active, axis=1, keepdims=True) + 1e-10)
    
    # 3. Two-NN Calculation
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', n_jobs=-1).fit(X_active)
    distances, _ = nbrs.kneighbors(X_active)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Filter out exact duplicates (where r1=0) to avoid division by zero
    mask_nonzero = (r1 > 1e-7)
    r1, r2 = r1[mask_nonzero], r2[mask_nonzero]
    
    mu = r2 / r1
    # Only keep mu values that are mathematically sound (>1)
    mu = mu[mu > 1.0001]
    
    if len(mu) < 500:
        return np.nan
    
    # 4. Robust Regression Fit
    n = len(mu)
    mu_sorted = np.sort(mu)
    # Empirical CDF
    f_empirical = np.arange(1, n + 1) / n
    
    # The Two-NN formula: d = log(1-F(mu)) / -log(mu)
    # We fit: y = d * x
    x_coords = np.log(mu_sorted)
    y_coords = -np.log(1 - f_empirical + 1e-10)
    
    # Use the 10th to 90th percentile for the fit to avoid tail noise
    p10, p90 = int(n*0.1), int(n*0.9)
    x_fit = x_coords[p10:p90]
    y_fit = y_coords[p10:p90]
    
    # Fit line through origin (0,0) as per Two-NN theory
    # d = (x'y) / (x'x)
    d = np.sum(x_fit * y_fit) / np.sum(x_fit**2)
    
    return d

# --- Configuration ---
#target_layer = 25
#target_layer = 27

for config in configs:
    print(f"Processing {config['MODEL']}")
    
    id_path=os.path.join(config["BASE_DIR"], "ID_result")
    if not os.path.exists(id_path):
        os.makedirs(id_path)

    for target_layer in config["LAYERS"]:
        root_path = os.path.join(config["BASE_DIR"], f"./hidden_states_{target_layer:02d}")
        results = [] # <--- Added initialization

        # Check if directory exists
        if not os.path.exists(root_path):
            print(f"Error: Path {root_path} does not exist.")
            exit()

        languages = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
        ### Added to remove birdsong and mls_spanish
        try:
            languages.remove("birdsong")
        except:
            pass
        try:
            languages.remove("mls_spanish")
        except:
            pass
        ### Added to remove birdsong and mls_spanish
        
        print(f"Starting ID Analysis for Layer {target_layer}...")

        for lang in languages:
            print(f"Analysing: {lang} ...")
            folder_name = f"hiddenlayer={target_layer}"
            file_path = os.path.join(root_path, lang, folder_name, "X_flat_stable_1hours.npy")
            
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path, mmap_mode='r')
                    
                    if len(data) > 30000:
                        idx = np.random.choice(len(data), 30000, replace=False)
                        data_sample = np.array(data[idx])
                    else:
                        data_sample = np.array(data)
                    
                    id_val = estimate_intrinsic_dimension(data_sample)
                    
                    if not np.isnan(id_val):
                        print(f"[SUCCESS] {lang:20} | ID: {id_val:.2f}")
                        results.append({"Language": lang, "Intrinsic_Dim": id_val})
                    else:
                        print(f"[WARNING] {lang:20} | ID resulted in NaN")
                        
                except Exception as e:
                    print(f"[ERROR]   {lang:20} | Could not process: {e}")
            else:
                print(f"[SKIP]    {lang:20} | Path not found.")

        # Save and Display
        if results:
            df_results = pd.DataFrame(results).sort_values(by="Intrinsic_Dim", ascending=False)
            output_file = os.path.join(id_path, f"ID_Analysis_L{target_layer}_Fixed.csv")
            df_results.to_csv(output_file, index=False)
            print(f"\nAnalysis complete. Results saved to {output_file}")
            print(df_results.head(10))
        else:
            print("\nNo results were generated.")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()
