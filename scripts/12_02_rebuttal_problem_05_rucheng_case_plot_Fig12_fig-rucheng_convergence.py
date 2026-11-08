import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def load_rucheng_convergence(id_path, f0_path, tda_path, total_layers):
    # 1. Load Structural ID Data for Rucheng
    id_list = []
    for layer in range(total_layers + 1):
        f = os.path.join(id_path, f"ID_Analysis_L{layer}_Fixed.csv")
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            # Match Language name precisely
            rucheng_id = temp_df[temp_df['Language'].str.contains('rucheng', case=False)]
            if not rucheng_id.empty:
                id_list.append({'Layer': layer, 'Mean_ID': rucheng_id['Intrinsic_Dim'].mean()})
    df_id = pd.DataFrame(id_list)
    
    # 2. Load Functional F0 Data for Rucheng
    df_f0 = pd.read_csv(f0_path)
    df_ru_f0 = df_f0[df_f0['Language'].str.contains('rucheng', case=False)].copy()
    df_ru_f0['Tone_Expertise'] = df_ru_f0['Corr_Height'].abs()
    df_f0_agg = df_ru_f0.groupby('Layer')['Tone_Expertise'].mean().reset_index()
    
    # 3. Load Topological TDA Data
    df_tda = pd.read_csv(tda_path)
    
    # Merge and Sort
    merged = pd.merge(df_id, df_f0_agg, on='Layer', how='inner')
    merged = pd.merge(merged, df_tda, on='Layer', how='inner')
    merged = merged.sort_values('Layer')
    
    return merged

# Configuration
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Load Data
data_rucheng = load_rucheng_convergence(
    os.path.join(configs[0]["BASE_DIR"], "ID_result/"), 
    os.path.join(configs[0]["BASE_DIR"], "global_results/Layer_Evolution_Analysis_allLayers.csv"),
    os.path.join(configs[0]["BASE_DIR"], "Rucheng_TDA_Evolution_Checkpointed.csv"), 
    len(configs[0]["LAYERS"])-1
)

# Plotting the Triple-Axis Convergence
fig, ax1 = plt.subplots(figsize=(8, 5))

# Axis 1: Structural ID (Shaded Area)
color_id = '#1f77b4'
ax1.fill_between(data_rucheng['Layer'], data_rucheng['Mean_ID'], color=color_id, alpha=0.15, label='Structural ID ($id$)')
ax1.plot(data_rucheng['Layer'], data_rucheng['Mean_ID'], color=color_id, lw=2, alpha=0.5)
ax1.set_xlabel("Layer", fontsize=11)
ax1.set_ylabel("Intrinsic Dimension ($id$)", color=color_id, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_id)
ax1.set_ylim(0, data_rucheng['Mean_ID'].max() * 1.1)

# Axis 2: Functional Expertise & Topological Logic
ax2 = ax1.twinx()

# Red: Functional Expertise
color_exp = '#d62728'
ax2.plot(data_rucheng['Layer'], data_rucheng['Tone_Expertise'], color=color_exp, lw=3, 
         label='Tone Expertise (Corr)')

# Green: Topological Logic (H1 Persistence)
color_tda = '#2ca02c'
ax2.plot(data_rucheng['Layer'], data_rucheng['Normalized_H1_Persistence'], color=color_tda, lw=3, 
         ls='--', label='Topological Logic ($H_1$)')

ax2.set_ylabel("Expertise / Persistence Score", color='black', fontsize=11)
ax2.set_ylim(0, 1.05)

# Shading the Crystallization Zone
plt.axvspan(33, 48, color='gray', alpha=0.1, label='Crystallization Zone')

# Styling and Grid
plt.title("Rucheng Tone Specialization: Structural, Functional, and Topological Convergence", 
          fontsize=12, fontweight='bold', pad=15)
ax1.grid(True, linestyle=':', alpha=0.6)

# Consolidated Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig("rucheng_convergence.pdf", bbox_inches='tight')
plt.savefig("rucheng_convergence.png", bbox_inches='tight')

# Correlation Stat
cryst_zone = data_rucheng[data_rucheng['Layer'] >= 33]
corr_val = cryst_zone['Tone_Expertise'].corr(cryst_zone['Normalized_H1_Persistence'])
print(f"Topological-Functional Correlation (L33-L48): {corr_val:.4f}")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()