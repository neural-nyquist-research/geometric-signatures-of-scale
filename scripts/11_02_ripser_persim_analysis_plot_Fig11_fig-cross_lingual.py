import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

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

# --- 1. Configuration ---
CSV_INPUT = os.path.join(configs[0]["BASE_DIR"], "topology_quant_results.csv")
LAYER = 38  # Key analysis layer
METRIC = 'total_topo_score'  # Can also choose 'topo_dist_h1'

# --- 2. Matrix Construction ---
df = pd.read_csv(CSV_INPUT)
ldf = df[df['layer'] == LAYER]

langs = sorted(list(set(ldf['lang_a']).union(set(ldf['lang_b']))))

# --- NEW TREATMENT CODE: Professional Labels ---
display_langs = []
for l in langs:
    if l == "_hn_rucheng":
        display_langs.append("Rucheng")
    else:
        # 1. Start by making it look like 'Italian_fleurs'
        clean_name = l.capitalize() 
        # 2. Replace the lowercase suffix with uppercase + space
        clean_name = clean_name.replace("_fleurs", " FLEURS").replace("fleurs", " FLEURS")
        display_langs.append(clean_name.strip())
# ------------------------------------------------

n = len(langs)
lmap = {l: i for i, l in enumerate(langs)}

dist_m = np.zeros((n, n))
for _, r in ldf.iterrows():
    i, j = lmap[r['lang_a']], lmap[r['lang_b']]
    dist_m[i, j] = dist_m[j, i] = r[METRIC]

# [Save intermediate matrix] for convenient cross-script calls or statistical analysis later
np.save(os.path.join(configs[0]["BASE_DIR"], f"dist_matrix_L{LAYER}.npy"), dist_m)
with open(os.path.join(configs[0]["BASE_DIR"], f"langs_list_L{LAYER}.txt"), "w") as f:
    f.write("\n".join(langs))

# --- 3. Clustering and Plotting ---
Z = linkage(squareform(dist_m), method='ward')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=150)

# A. Heatmap - Use display_langs for ticks
sns.heatmap(dist_m, xticklabels=display_langs, yticklabels=display_langs, cmap="rocket_r", ax=ax1)

# Set size and rotation for x-axis
ax1.set_xticklabels(ax1.get_xticklabels(), size=14, rotation=90)
ax1.set_yticklabels(ax1.get_yticklabels(), size=14)
ax1.set_title(f"Topological Distance Heatmap (Layer {LAYER})", fontsize=15)

# B. Dendrogram - Use display_langs for labels
dendrogram(
    Z, 
    labels=display_langs, # Updated here
    leaf_rotation=90, 
    leaf_font_size=14, 
    ax=ax2
)

ax2.set_title(f"Topological Dendrogram (Layer {LAYER})", fontsize=15)

plt.tight_layout()
plt.savefig(f"Final_Analysis_L{LAYER}.png")
plt.show()

# --- 4. Automatic Clustering Suggestions ---
# Assume we want to divide 24 languages into 5 major topological groups
clusters = fcluster(Z, t=5, criterion='maxclust')
cluster_res = pd.DataFrame({'Language': langs, 'Cluster': clusters})
cluster_res.to_csv(f"language_clusters_L{LAYER}.csv", index=False)
print(f"✅ Analysis for Layer {LAYER} completed; suggested clustering results saved.")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()