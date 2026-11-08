#pip install ripser persim

import random
import numpy as np
import pandas as pd
import pickle
import umap
from ripser import ripser
from persim import wasserstein
from tqdm import tqdm
from itertools import combinations
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
    },
]

def global_seed_lock(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If deep learning frameworks (such as PyTorch) are involved later, they also need to be locked here
    print(f"🔒 Global random seed has been locked to: {seed}")

global_seed_lock(42)

# --- 1. Configuration paths ---
BASE_PATH = configs[0]["BASE_DIR"]
FULL_CLOUD_CACHE = os.path.join(configs[0]["BASE_DIR"], "umap_full_cloud_cache")  # New directory for full point clouds
TOPO_CACHE = os.path.join(configs[0]["BASE_DIR"], "topology_signatures")          # Persistence diagrams (fingerprints) directory
RESULT_CSV = os.path.join(configs[0]["BASE_DIR"], "topology_quant_results.csv")
TARGET_LAYERS = configs[0]["LAYERS"]

os.makedirs(FULL_CLOUD_CACHE, exist_ok=True)
os.makedirs(TOPO_CACHE, exist_ok=True)

class TopologyPipeline:
    def __init__(self):
        self.reducer = umap.UMAP(n_components=3, metric='cosine', random_state=42)

    def get_raw_data_path(self, layer, lang):
        """Match your hidden_states directory structure"""
        sub_dir = f"hiddenlayer={layer}"
        return os.path.join(BASE_PATH, f"hidden_states_{layer:02d}", lang, sub_dir, "X_flat_stable_1hours.npy")

    def get_full_cloud(self, layer, lang):
        """Get or compute the full 3D point cloud cache"""
        cache_path = os.path.join(FULL_CLOUD_CACHE, f"cloud_L{layer:02d}_{lang}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        # Load high-dimensional raw data
        raw_path = self.get_raw_data_path(layer, lang)
        if not os.path.exists(raw_path): return None
        
        raw_data = np.load(raw_path)
        # Assume original data is (N, Dim); reshape if it's 1D
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(-1, 1024) # Assume dimension is 1024; please modify according to actual data

        # UMAP dimensionality reduction (preserving topology of the full point cloud)
        print(f"Reducing dimensions: L{layer} - {lang} (Points: {len(raw_data)})")
        cloud_3d = self.reducer.fit_transform(raw_data)
        
        # Normalize (Z-score) to ensure different languages are compared on the same scale for topology
        cloud_3d = StandardScaler().fit_transform(cloud_3d)
        
        np.save(cache_path, cloud_3d)
        return cloud_3d

    def get_persistence_dgm(self, layer, lang, seed=42):
            """Get or compute the topological fingerprint (Persistence Diagram)"""
            cache_path = os.path.join(TOPO_CACHE, f"dgm_L{layer:02d}_{lang}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f: 
                    return pickle.load(f)

            points = self.get_full_cloud(layer, lang)
            if points is None: 
                return None

            # --- Core randomness control area ---
            if len(points) > 1500:
                # Critical: Fix numpy's random seed before sampling
                # Ensure that no matter how many times you run it, the point cloud subset extracted for the same layer and language is exactly identical
                np.random.seed(seed) 
                idx = np.random.choice(len(points), 1500, replace=False)
                points = points[idx]
            # -----------------------

            # Compute persistence homology in H0 and H1 dimensions
            # If ripser is based on distance matrices internally, the result will be deterministic
            dgms = ripser(points, maxdim=1)['dgms']
            
            with open(cache_path, 'wb') as f:
                pickle.dump(dgms, f)
            return dgms

    def run_full_quant(self):
        all_results = []
        if os.path.exists(RESULT_CSV):
            all_results = pd.read_csv(RESULT_CSV).to_dict('records')

        for layer in TARGET_LAYERS:
            # Automatically detect which languages exist in the current layer
            layer_path = os.path.join(BASE_PATH, f"hidden_states_{layer:02d}")
            if not os.path.exists(layer_path): continue
            langs = [d for d in os.listdir(layer_path) if os.path.isdir(os.path.join(layer_path, d))]
            ### Added to remove birdsong and mls_spanish
            try:
                langs.remove("birdsong")
            except:
                pass
            try:
                langs.remove("mls_spanish")
            except:
                pass
            ### Added to remove birdsong and mls_spanish

            # Pairwise comparison
            for lang_a, lang_b in tqdm(combinations(langs, 2), desc=f"Layer {layer} Topology"):
                # Check if already computed
                exists = any(r for r in all_results if r['layer']==layer and 
                             set([r['lang_a'], r['lang_b']]) == set([lang_a, lang_b]))
                if exists: continue

                dgm_a = self.get_persistence_dgm(layer, lang_a)
                dgm_b = self.get_persistence_dgm(layer, lang_b)

                if dgm_a is not None and dgm_b is not None:
                    # Calculate distance in H1 dimension (reflecting similarity of "loops" in manifold structure)
                    dist_h1 = wasserstein(dgm_a[1], dgm_b[1]) if len(dgm_a[1])>0 and len(dgm_b[1])>0 else 0
                    dist_h0 = wasserstein(dgm_a[0], dgm_b[0])
                    
                    all_results.append({
                        "layer": layer,
                        "lang_a": lang_a,
                        "lang_b": lang_b,
                        "topo_dist_h1": dist_h1,
                        "topo_dist_h0": dist_h0,
                        "total_topo_score": np.sqrt(dist_h1**2 + dist_h0**2)
                    })
                    # Save results in real-time
                    pd.DataFrame(all_results).to_csv(RESULT_CSV, index=False)

# --- Execution ---
pipeline = TopologyPipeline()
pipeline.run_full_quant()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()
