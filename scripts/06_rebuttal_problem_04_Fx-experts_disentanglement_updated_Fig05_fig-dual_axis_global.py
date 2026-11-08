import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_comprehensive_data(id_path, tone_path, formant_path, total_layers):
    # 1. Load Structural ID Data
    id_list = []
    for layer in range(total_layers + 1):
        f = os.path.join(id_path, f"ID_Analysis_L{layer}_Fixed.csv")
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            id_list.append({'Layer': layer, 'Mean_ID': temp_df['Intrinsic_Dim'].mean()})
    df_id = pd.DataFrame(id_list)
    
    # 2. Load Tone Expert Data
    df_tone = pd.read_csv(tone_path)
    df_tone['Tone_Expertise'] = df_tone['Corr_Height'].abs()
    df_tone_agg = df_tone.groupby('Layer')['Tone_Expertise'].mean().reset_index()
    
    # 3. Load Formant Expert Data
    df_formant = pd.read_csv(formant_path)
    df_formant['Formant_Expertise'] = df_formant['Max_Corr'].abs()
    # Average across all formants (F1, F2, F3) to get a 'Spectral Mastery' score
    df_formant_agg = df_formant.groupby('Layer')['Formant_Expertise'].mean().reset_index()
    
    # Merge all
    merged = pd.merge(df_id, df_tone_agg, on='Layer')
    merged = pd.merge(merged, df_formant_agg, on='Layer')
    merged['Rel_Depth'] = merged['Layer'] / total_layers
    return merged

# Load Both Models
data_2b = load_comprehensive_data(os.path.join(configs[0]["BASE_DIR"], "ID_result"), 
                                  f"{configs[0]["BASE_DIR"]}global_results/Layer_Evolution_Analysis_allLayers.csv",
                                  configs[0]["csv_path"], len(configs[0]["LAYERS"])-1)

data_1b = load_comprehensive_data(os.path.join(configs[1]["BASE_DIR"], "ID_result"), 
                                  f"{configs[1]["BASE_DIR"]}global_results/Layer_Evolution_Analysis_allLayers.csv",
                                  configs[1]["csv_path"], len(configs[1]["LAYERS"])-1)

data_53 = load_comprehensive_data(os.path.join(configs[2]["BASE_DIR"], "ID_result"), 
                                  f"{configs[2]["BASE_DIR"]}global_results/Layer_Evolution_Analysis_allLayers.csv",
                                  configs[2]["csv_path"], len(configs[2]["LAYERS"])-1)

data_300 = load_comprehensive_data(os.path.join(configs[3]["BASE_DIR"], "ID_result"), 
                                  f"{configs[3]["BASE_DIR"]}global_results/Layer_Evolution_Analysis_allLayers.csv",
                                  configs[3]["csv_path"], len(configs[3]["LAYERS"])-1)

# Visualization
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17, 5))

def plot_triple_axis(ax_main, data, title):
    # ID - The Structural Reservoir
    ax_main.plot(data['Rel_Depth'], data['Mean_ID'], color='gray', lw=4, alpha=0.3, label='ID ($d$)')
    ax_main.set_ylabel("Intrinsic Dimension ($id$)", color='gray')
    
    # Functional Experts
    ax_func = ax_main.twinx()
    ax_func.plot(data['Rel_Depth'], data['Formant_Expertise'], color='#1f77b4', lw=3, label='Formant (Segmental)')
    ax_func.plot(data['Rel_Depth'], data['Tone_Expertise'], color='#d62728', lw=3, label='Tone (Suprasegmental)')
    ax_func.set_ylabel("Correlation (Expertise)")
    
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.set_xlabel("Relative Depth")
    ax_func.legend(loc='upper left')

plot_triple_axis(ax1, data_53, "XLSR-53: Congested Processing")
plot_triple_axis(ax2, data_300, "XLS-R-300: Congested Processing")
plot_triple_axis(ax3, data_1b, "XLS-R-1B: Sequential Specialization")
plot_triple_axis(ax4, data_2b, "XLS-R-2B: Sequential Specialization")

plt.tight_layout()
plt.savefig("Functional_Hierarchy_Segmental_vs_Suprasegmental.png", dpi=300)
plt.show()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()