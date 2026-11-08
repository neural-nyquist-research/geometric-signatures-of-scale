import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

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

def load_id_data(base_path, total_layers, model_name):
    all_data = []
    for layer in range(total_layers + 1):
        file_path = os.path.join(base_path, f"ID_Analysis_L{layer}_Fixed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Layer'] = layer
            df['Relative_Depth'] = layer / total_layers
            df['Model'] = model_name
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# 1. LOAD AND PREPARE DATA
path_2b = os.path.join(configs[0]["BASE_DIR"], "ID_result/")
path_53 = os.path.join(configs[2]["BASE_DIR"], "ID_result/")

df_2b = load_id_data(path_2b, len(configs[0]["LAYERS"])-1, "XLS-R-2B")
df_53 = load_id_data(path_53, len(configs[2]["LAYERS"])-1, "XLSR-53")

# Language Name Normalization Mapping
name_mapping = {
    "_hn_rucheng": "Rucheng",
    "english_fleurs": "English_FLEURS",
    "cantonese_fleurs": "Cantonese_FLEURS"
}

# Apply normalization and create the combined dataframe
for df in [df_2b, df_53]:
    if not df.empty:
        df['Language'] = df['Language'].replace(name_mapping)

combined_df = pd.concat([df_2b, df_53], ignore_index=True)

# 2. SETUP PLOT
plt.figure(figsize=(14, 8))
sns.set_style("white")
palette = {"XLS-R-2B": "#1f77b4", "XLSR-53": "#ff7f0e"}

# 3. PLOT MEAN TRENDS WITH SD SHADING
sns.lineplot(
    data=combined_df, x='Relative_Depth', y='Intrinsic_Dim', 
    hue='Model', palette=palette, errorbar='sd', linewidth=4, alpha=0.8, zorder=5
)

# 4. PLOT SPECIFIC LANGUAGES
target_langs = ["Rucheng", "Cantonese_FLEURS", "English_FLEURS"] 
lang_colors = ["#d62728", "#2ca02c", "#9467bd"] # Red, Green, Purple

for lang, l_color in zip(target_langs, lang_colors):
    # XLS-R-2B paths (solid)
    lang_data_2b = df_2b[df_2b['Language'] == lang]
    if not lang_data_2b.empty:
        plt.plot(lang_data_2b['Relative_Depth'], lang_data_2b['Intrinsic_Dim'], 
                 color=l_color, linewidth=1.5, linestyle='-', alpha=0.8, zorder=10)
    
    # XLSR-53 paths (dashed)
    lang_data_53 = df_53[df_53['Language'] == lang]
    if not lang_data_53.empty:
        plt.plot(lang_data_53['Relative_Depth'], lang_data_53['Intrinsic_Dim'], 
                 color=l_color, linewidth=1.5, linestyle='--', alpha=0.6, zorder=8)

# 5. ANNOTATIONS
# XLSR-53 Premature Peak (~0.55 depth)
plt.annotate('Premature Crystallization Peak', 
             xy=(0.58, 15.6), xytext=(0.2, 13),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', color=palette["XLSR-53"])

# XLS-R-2B Delayed Peak (~0.7 depth)
plt.annotate('Delayed Structural Peak\n(Topological Safety Margin)', 
             xy=(0.68, 17.7), xytext=(0.4, 19),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', color=palette["XLS-R-2B"])

# Crystallization Zone Shading
plt.axvspan(0.8, 1.0, color='gray', alpha=0.1, zorder=1)
plt.text(0.9, 11, 'Crystallization\nZone', horizontalalignment='center', 
         fontsize=12, style='italic', alpha=0.7)

# 6. FINAL STYLING
plt.title("Intrinsic Dimension (ID) Trajectories: Model Trends vs. Individual Languages", fontsize=16, fontweight='bold')
plt.xlabel("Normalized Relative Depth (0.0 = Input, 1.0 = Output)", fontsize=14)
plt.ylabel("Intrinsic Dimension ($id$)", fontsize=14)
plt.ylim(10, 20)

# Clean Legend
legend_elements = [
    Line2D([0], [0], color=palette["XLS-R-2B"], lw=4, label='XLS-R-2B (Mean Trend)'),
    Line2D([0], [0], color=palette["XLSR-53"], lw=4, label='XLSR-53 (Mean Trend)'),
    Line2D([0], [0], color='black', lw=1.5, ls='-', label='2B Language Path'),
    Line2D([0], [0], color='black', lw=1.5, ls='--', label='53 Language Path'),
    Line2D([0], [0], marker='o', color='w', label='Rucheng', markerfacecolor=lang_colors[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Cantonese', markerfacecolor=lang_colors[1], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='English', markerfacecolor=lang_colors[2], markersize=10),
]
plt.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/fig5_id_language_paths.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()