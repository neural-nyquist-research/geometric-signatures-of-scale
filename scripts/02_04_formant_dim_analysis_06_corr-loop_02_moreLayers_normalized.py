import pandas as pd
import numpy as np
from ripser import ripser
from scipy.spatial import ConvexHull
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

def calculate_dispersion(df, layer_val):
    points = df[df['Layer'] == layer_val][['Max_Corr', 'Best_Lag_Sec']].values
    if len(points) < 3: return 0
    hull = ConvexHull(points)
    return hull.volume # In 2D, volume represents area



def analyze_layer_topology(df, layer_val, persistence_thresh=0.08):
    """
    Calculate the Betti-1 number (number of logical loops) for a specific layer
    """
    # Extract (Max_Corr, Best_Lag_Sec) as point cloud coordinates
    points = df[df['Layer'] == layer_val][['Max_Corr', 'Best_Lag_Sec']].values
    
    if len(points) < 5:
        return 0, len(points)

    # Calculate persistent homology (maxdim=1 indicates calculation of H1 loops)
    results = ripser(points, maxdim=1)
    h1_diagram = results['dgms'][1] 
    
    if len(h1_diagram) == 0:
        return 0, len(points)
    
    # Filter for significant persistence (Death - Birth > threshold)
    persistence = h1_diagram[:, 1] - h1_diagram[:, 0]
    significant_loops = np.sum(persistence > persistence_thresh)
    
    return int(significant_loops), len(points)


def calculate_dispersion_normalized(df, layer_val, total_model_dim):
    """
    Calculate the normalized manifold coverage area
    total_model_dim: 53 model is 1024, 2B model is 1920
    """
    points = df[df['Layer'] == layer_val][['Max_Corr', 'Best_Lag_Sec']].values
    if len(points) < 3: 
        return 0, 0

    # --- degeneracy guard: need non-collinear points for a 2-D hull ---
    # Rank of the centered point matrix: <2 means all points lie on a line
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        # flat manifold: physical area is 0 (it's a line), normalize still 0
        return 0, 0.0

    # Calculate original convex hull area
    try:
        hull = ConvexHull(points)
    except Exception:
        # fallback for any other numerical degeneracy
        return 0, 0.0

    raw_area = hull.volume          # area in 2-D
    
    # Approach A: Normalization based on current significant node count (reflects distribution efficiency)
    node_count = len(points)
    norm_area = raw_area / np.log2(node_count)
    
    # Approach B: Normalization based on total model dimension (reflects overall capacity utilization - recommended to use with Entropy)
    # norm_area = raw_area / np.log2(total_model_dim)
    
    return norm_area, raw_area


results_csv=[]
for config in configs:
    print(f"Processing: {config["MODEL"]}")
    # ==========================================
    # Step 1: Acquire significant_df
    # ==========================================

    # Read and filter: retain only dimensions with absolute correlation > 0.1
    df_raw = pd.read_csv(config["csv_path"])
    significant_df = df_raw[df_raw['Max_Corr'].abs() > 0.1].copy()

    print(f"Successfully loaded {config["MODEL"]} data, totaling {len(significant_df)} significant dimensions.")

    # ==========================================
    # Step 2: Connect topology analysis logic
    # ==========================================

    topo_results = []
    #for l in [4, 12, 19]:
    for l in config["LAYERS"]:
        b1_count, node_count = analyze_layer_topology(significant_df, l)
        topo_results.append({
            'Layer': l,
            'Betti_1': b1_count,
            'Node_Count': node_count,
            'Model': config["MODEL"]
        })
        
        norm, area = calculate_dispersion_normalized(significant_df, l, total_model_dim=config["DIM_SIZE"])
        print(f"{config["MODEL"]} (L{l}) Physical coverage area: {area:.4f}; Normalized area: {norm:.4f}")
        
        results_csv.append({"Model": config['MODEL'],
                            "Layer": l,
                            "Physical Coverage Area": area,
                            "Normalized Manifold Area": norm,
        })

    df_topo = pd.DataFrame(topo_results)
    print(f"\n### {config['MODEL']} Topology Feature Calculation Results ###")
    print(df_topo)

pd.DataFrame(results_csv).to_csv(r"./re-organized/manifold_area.csv", index=False)

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()
