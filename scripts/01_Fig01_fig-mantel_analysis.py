import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
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


# --- Color definitions added at the top of the script or before plotting logic ---
GRAY_MID = "#8B949E"  # Classic academic medium gray
GRAY_LIGHT = "#D1D5DA"
RED_ALPHA = "#FF7B721A" # Red with transparency, used for semantic erasure zones

# --- 1. Configuration and Switches ---
BASE_DATA_DIR = "./re-organized/wav2vec2-xls-r-2b/"
TYPOLOGY_CSV = "./cldf-datasets_wals/target_languages_typology.csv"
CACHE_FILE = "mantel_results_cache.csv"  # Results cache file
LAYERS = [f"{i:02d}" for i in range(49)] 

folder_to_iso = {
    'arabic': 'arb', 'cantonese': 'yue', 'dutch': 'nld', 'english': 'eng',
    'estonian': 'ekk', 'finnish': 'fin', 'french': 'fra', 'georgian': 'kat',
    'german': 'deu', 'indonesian': 'ind', 'italian': 'ita', 'japanese': 'jpn',
    'korean': 'kor', 'mandarin': 'cmn', 'polish': 'pol', 'portuguese': 'por',
    'russian': 'rus', 'spanish': 'spa', 'swahili': 'swh', 'tamil': 'tam',
    'thai': 'tha', 'turkish': 'tur', 'vietnamese': 'vie', 'zulu': 'zul'
}

feature_cols = ['Syllable Structure', 'Tone', 'Consonant Inventories', 'Vowel Quality Inventories', 'voiced_ratio']

# --- 2. Pre-computation Functions (Called only if no cache exists) ---
def precompute_layer_prototypes(df_typeo):
    layer_cache = {}
    print(f">>> Reading raw data and computing language prototypes...")
    for l_str in tqdm(LAYERS, desc="Layers"):
        layer_dir = os.path.join(BASE_DATA_DIR, f"hidden_states_{l_str}")
        protos_raw, feats_raw = [], []
        
        for folder_prefix, iso in tqdm(folder_to_iso.items(), desc="Languages", leave=False):
            path_part = f"hiddenlayer={int(l_str)}"
            file_path = os.path.join(layer_dir, f"{folder_prefix}_fleurs", path_part, "X_flat_stable_1hours.npy")
            
            if os.path.exists(file_path):
                # mmap_mode saves 16GB memory pressure
                data = np.load(file_path, mmap_mode='r')
                proto = np.mean(data, axis=0)
                
                typeo_data = df_typeo[df_typeo['ISO639P3code'] == iso][feature_cols].values
                if len(typeo_data) > 0:
                    protos_raw.append(proto)
                    feats_raw.append(typeo_data[0])
                del data
        
        layer_cache[l_str] = {'protos': np.array(protos_raw), 'feats': np.array(feats_raw)}
        gc.collect()
    return layer_cache

# --- 3. Main Logic: Check Cache or Recompute ---
try:
    if os.path.exists(CACHE_FILE):
        print(f"Found cached results file: {CACHE_FILE}, reading directly for plotting...")
        res_df = pd.read_csv(CACHE_FILE)
    else:
        print("Cache file not found, starting scan and computation...")
        df_typeo = pd.read_csv(TYPOLOGY_CSV)
        df_typeo['ISO639P3code'] = df_typeo['ISO639P3code'].str.strip()
        
        layer_data_cache = precompute_layer_prototypes(df_typeo)
        
        all_results = [] # Defined only in computation mode
        for use_l2 in tqdm([True, False], desc="L2Norm"):
            for metric in tqdm(['cosine', 'euclidean'], desc="Distance", leave=False):
                label = f"Norm={use_l2}, Metric={metric.capitalize()}"
                print(f"Analyzing: {label}")
                for l_str in tqdm(LAYERS, desc="Layers", leave=False):
                    d = layer_data_cache[l_str]
                    if len(d['protos']) < 10: continue
                    p_work = d['protos'].copy()
                    if use_l2: p_work = normalize(p_work, axis=1, norm='l2')
                    
                    d_speech = squareform(pdist(p_work, metric=metric))
                    d_typeo = squareform(pdist(d['feats'], metric='euclidean'))
                    idx = np.triu_indices(len(d_speech), k=1)
                    r, _ = pearsonr(d_speech[idx], d_typeo[idx])
                    all_results.append({'Layer': int(l_str), 'Mantel_R': r, 'Label': label})
        
        res_df = pd.DataFrame(all_results)
        res_df.to_csv(CACHE_FILE, index=False)
        print(f"Computation complete, results saved to: {CACHE_FILE}")
        
# --- 4. Plotting Analysis and Auto-Annotation (Compatibility Fixed Version) ---
    try:
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

        # Select target curves for annotation
        target_label = "Norm=True, Metric=Cosine"
        if target_label in res_df['Label'].unique():
            best_curve = res_df[res_df['Label'] == target_label]
        else:
            best_curve = res_df[res_df['Label'] == res_df['Label'].unique()[0]]

        # Plot all curves
        plot = sns.lineplot(
            data=res_df, x='Layer', y='Mantel_R', hue='Label', 
            style='Label', markers=True, dashes=[(None, None), (2, 2), (5, 5), (1, 1)], 
            linewidth=2.5, alpha=0.8
        )


        # --- Redefined annotation points for XLS-R-2B ---
        # 1. L6: Initial encoding 
        # 2. L16: Preliminary fusion of phonology and multimodal features 
        # 3. L22: Entry into long-range stable plateau 
        # 4. L38: Global typological peak (Zenith)
        annotations = [
            (6, "Early Acoustic Encoding", "top"),
            (16, "Syllabic-Phonetic Integration", "bottom"), 
            (22, "Structural Stability Entry", "top"), 
            (38, "Global Typological Zenith", "top") 
        ]

        for l_idx, text, pos in annotations:
            row = best_curve[best_curve['Layer'] == l_idx]
            if not row.empty:
                r_val = row['Mantel_R'].values[0]
                # Fine-tune offset based on higher r values in 2B model
                offset = 0.05 if pos == "top" else -0.06
                va = 'bottom' if pos == "top" else 'top'
                
                plt.annotate(
                    text, xy=(l_idx, r_val), xytext=(l_idx, r_val + offset),
                    arrowprops=dict(arrowstyle='-|>', color=GRAY_MID, lw=1.5),
                    fontsize=12, fontweight='bold', ha='center', va=va,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRAY_MID, alpha=0.9, lw=1)
                )

        # --- 1. Blue trend guide line: Covering the ultra-long "plateau" of the 2B model ---
        # The characteristic of the 2B model is a very stable region between L24-L40, called Sustained Typological Retention
        plt.annotate("Sustained Typological Retention", 
                     xy=(32, 0.45), xytext=(24, 0.58),
                     arrowprops=dict(arrowstyle='->', color='#2188ff', lw=2, ls='--'),
                     fontsize=12, color='#0366d6', fontweight='bold', ha='center')
                     
        # --- 2. Red area adjustment: "Semantic erasure" in the 2B model happens very late ---
        # According to the chart, the cliff drop starts after L44, so set red mask from 44-48
        plt.axvspan(44, 48, color='#FF7B72', alpha=0.15, label='Discrete Unit Mapping Zone')
        
        # Place text near L46 where the drop is most severe
        plt.text(46, 0.20, "Rapid\nInformation\nCollapse", 
                 color='#D73A49', ha='center', va='center', fontsize=12, fontweight='bold')

        # --- 3. Axis optimization ---
        # The r values for the 2B model are very high (close to 0.5), so y-axis range needs adjustment
        plt.ylim(-0.15, 0.70)
        
        plt.title("Sustained Typological Awareness in XLS-R-2B (48-Layer)\nCross-layer Mantel Correlation Analysis", 
                  fontsize=16, pad=25, fontweight='bold')
        plt.xlabel("Transformer Layer Index (Encoder)", fontsize=12)
        plt.ylabel("Typological Correlation (Mantel $r$)", fontsize=12)
        
        # The 2B model has many layers; label every 2nd layer or all with smaller font size
        plt.xticks(range(0, 49, 2), fontsize=9)


        
        # Optimize axis range to prevent top annotations from being cut off
        y_max = res_df['Mantel_R'].max()
        plt.ylim(res_df['Mantel_R'].min() - 0.1, y_max + 0.15) 

        plt.legend(title="Extraction Parameters", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(r"./re-organized/xls-r-2b_typology_final.png", dpi=300, bbox_inches='tight') # Changed to png for quick viewing
        plt.show()

    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
except Exception as e:
    import traceback
    traceback.print_exc()
    
print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()