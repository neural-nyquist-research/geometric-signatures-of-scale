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
def load_data(id_path, f0_path, total_layers):
    # 1. Load ID Data
    id_list = []
    for layer in range(total_layers + 1):
        f = os.path.join(id_path, f"ID_Analysis_L{layer}_Fixed.csv")
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            id_list.append({'Layer': layer, 'Mean_ID': temp_df['Intrinsic_Dim'].mean()})
    df_id = pd.DataFrame(id_list)
    
    # 2. Load F0 Expert Data + Calculate Committee Variance
    df_f0 = pd.read_csv(f0_path)
    df_f0['Abs_Corr'] = df_f0['Corr_Height'].abs()
    
    # Group by Layer and Language first to get the variance of the 3 experts
    committee_stats = df_f0.groupby(['Layer', 'Language'])['Abs_Corr'].agg(['mean', 'std']).reset_index()
    # Then average across all languages for a global layer-wise trend
    df_exp = committee_stats.groupby('Layer')[['mean', 'std']].mean().reset_index()
    df_exp.columns = ['Layer', 'Expertise_Mean', 'Expertise_Std']
    
    # 3. Merge and Normalize
    merged = pd.merge(df_id, df_exp, on='Layer')
    merged['Rel_Depth'] = merged['Layer'] / total_layers
    return merged

# Load Both Models
data_2b = load_data(os.path.join(configs[0]["BASE_DIR"], "ID_result"), os.path.join(configs[0]["BASE_DIR"], "global_results/Layer_Evolution_Analysis_allLayers.csv"), len(configs[0]["LAYERS"])-1)
data_1b = load_data(os.path.join(configs[1]["BASE_DIR"], "ID_result"), os.path.join(configs[1]["BASE_DIR"], "global_results/Layer_Evolution_Analysis_allLayers.csv"), len(configs[1]["LAYERS"])-1)
data_53 = load_data(os.path.join(configs[2]["BASE_DIR"], "ID_result"), os.path.join(configs[2]["BASE_DIR"], "global_results/Layer_Evolution_Analysis_allLayers.csv"), len(configs[2]["LAYERS"])-1)
data_300 = load_data(os.path.join(configs[3]["BASE_DIR"], "ID_result"), os.path.join(configs[3]["BASE_DIR"], "global_results/Layer_Evolution_Analysis_allLayers.csv"), len(configs[3]["LAYERS"])-1)



# Visualization
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17, 5), sharey=False)

def plot_dual_axis(ax_main, data, title):
    color_id, color_exp = '#1f77b4', '#d62728'
    
    # 1. Structural ID (Left Axis)
    ax_main.plot(data['Rel_Depth'], data['Mean_ID'], color=color_id, lw=4, label='Mean ID ($d$)')
    ax_main.set_xlabel("Relative Depth", fontsize=12)
    ax_main.set_ylabel("Intrinsic Dimension ($id$)", color=color_id, fontsize=12)
    ax_main.tick_params(axis='y', labelcolor=color_id)
    
    # 2. Functional Expertise + Variance (Right Axis)
    ax_twin = ax_main.twinx()
    # Plot the Mean line
    ax_twin.plot(data['Rel_Depth'], data['Expertise_Mean'], color=color_exp, lw=3, ls='--', label='Tone Expertise')
    # Plot the Variance (Standard Deviation) Envelope
    ax_twin.fill_between(data['Rel_Depth'], 
                         data['Expertise_Mean'] - data['Expertise_Std'], 
                         data['Expertise_Mean'] + data['Expertise_Std'], 
                         color=color_exp, alpha=0.15, label='Committee Variance')
    
    ax_twin.set_ylabel("Mean $F_0$ Corr", color=color_exp, fontsize=12)
    ax_twin.tick_params(axis='y', labelcolor=color_exp)
    
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.grid(axis='y', alpha=0.3)

plot_dual_axis(ax1, data_53, "XLSR-53 (300M): Early Saturation")
plot_dual_axis(ax2, data_300, "XLS-R-300M (300M): Early Saturation")
plot_dual_axis(ax3, data_1b, "XLS-R-1B (1B): Late-Stage Specialization")
plot_dual_axis(ax4, data_2b, "XLS-R-2B (2B): Late-Stage Specialization")

# Legend logic: Combining labels from both axes for one legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1.get_shared_x_axes().get_siblings(ax1)[0].get_legend_handles_labels() # Logic for twin axis
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(r"./figures/fig3_functional_hierarchy.png", dpi=300)
plt.show()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()
    