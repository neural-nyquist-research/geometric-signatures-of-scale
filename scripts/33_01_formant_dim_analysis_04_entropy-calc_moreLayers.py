import pandas as pd
import numpy as np
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
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-2b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1920,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-2b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-xls-r-2b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLS-R-1B", 
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-1b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1280,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-1b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-xls-r-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLSR-53",
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-large-xlsr-53/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-large-xlsr-53/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-large-xlsr-53/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "XLS-R-300M",
        "BASE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-300m/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/wav2vec2-xls-r-300m/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/wav2vec2-xls-r-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "MMS-1B",
        "BASE_DIR" : r"./re-organized_mls/mms-1b/",
        "LAYERS" : list(range(49)),
        "DIM_SIZE" : 1280,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/mms-1b/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/mms-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "MMS-300M",
        "BASE_DIR" : r"./re-organized_mls/mms-300m/",
        "LAYERS" : list(range(25)),
        "DIM_SIZE" : 1024,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/mms-300m/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/mms-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
    {   "MODEL" : "mHuBERT-147",
        "BASE_DIR" : r"./re-organized_mls/mHuBERT-147/",
        "LAYERS" : list(range(13)),
        "DIM_SIZE" : 768,
        "TYPOLOGY_PATH" : r"./cldf-datasets_wals/target_languages_typology.csv",
        "CACHE_DIR" : r"./re-organized_mls/mHuBERT-147/formant_dim_analysis_cache",
        "csv_path" : r"./re-organized_mls/mHuBERT-147/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv",
    },
]


def calculate_temporal_entropy(df, bin_width=0.05, lag_range=(-1.5, 1.5)):
    """
    Calculate the temporal entropy for each layer.
    It is recommended to set bin_width finer (e.g., 0.05s) because the distribution of the 53 model is very concentrated.
    """
    bins = np.arange(lag_range[0], lag_range[1] + bin_width, bin_width)
    entropy_results = []
    
    # Calculate grouped by Layer
    for layer_val in df['Layer'].unique():
        # Extract all Lag data for the current layer (across languages, dimensions, and formants)
        layer_data = df[df['Layer'] == layer_val]['Best_Lag_Sec']
        
        # Statistics distribution
        counts, _ = np.histogram(layer_data, bins=bins)
        
        # Normalize to probability distribution
        probs = counts / counts.sum()
        probs = probs[probs > 0] # Remove zero terms
        
        # Calculate Shannon entropy (bits)
        entropy = -np.sum(probs * np.log2(probs))
        
        entropy_results.append({
            'Layer': f"L{layer_val}", 
            'Layer_Int': int(layer_val), 
            'Entropy': entropy
        })
        
    return pd.DataFrame(entropy_results).sort_values('Layer_Int')

# --- Execution Verification ---
for config in configs:

    # 1. Read data
    df_all = pd.read_csv(config["csv_path"]) 

    # 2. Core filtering: Recommended to keep only points with high correlation, excluding noise
    # For example, look only at dimensions where the absolute value of correlation > 0.1
    significant_df = df_all[df_all['Max_Corr'].abs() > 0.1].copy()

    # 3. Calculate entropy
    entropy_df = calculate_temporal_entropy(significant_df)

    # 4. Calculate relative depth (Set XLSR-53 to 24)
    TOTAL_LAYERS = len(config["LAYERS"])-1  
    entropy_df['Relative_Depth'] = entropy_df['Layer_Int'] / TOTAL_LAYERS

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot evolution curve
    sns.lineplot(data=entropy_df, x='Relative_Depth', y='Entropy', 
                 marker='o', markersize=8, color='#2E86C1', linewidth=2)

    # Decoration
    title=r'Temporal Entropy ($\mathcal{H}_T$) Evolution: '
    title+=f"{config["MODEL"]}"
    plt.title(title, fontsize=14)
    plt.xlabel('Relative Depth (Layer / Total)', fontsize=12)
    plt.ylabel('Temporal Entropy (Shannon Bits)', fontsize=12)
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Predictive Bottleneck (0.5 Depth)')

    # Annotate max entropy and min entropy points
    max_e = entropy_df.loc[entropy_df['Entropy'].idxmax()]
    plt.annotate(f'Max Entropy: {max_e.Entropy:.2f}', 
                 xy=(max_e.Relative_Depth, max_e.Entropy), 
                 xytext=(max_e.Relative_Depth+0.05, max_e.Entropy+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./re-organized_mls/{config['MODEL']}_Temporal_Entropy_Plot_moreLayers.png", dpi=300)
    plt.show()

    print(entropy_df)


    # =========================================================
    # 5. Automated data export logic (replaces the original print section)
    # =========================================================

    # --- Configuration Area: Only modify this when switching models ---
    MODEL_LABEL = config["MODEL"]  # If it is the 53 model, please change to "XLSR-53"

    # Add identifier column to entropy_df
    entropy_df['Model'] = MODEL_LABEL

    # A. Export trend data (for use in the top graph of a 2x1 combination plot)
    trend_filename = f"Entropy_Results_{MODEL_LABEL.replace('-', '_')}_moreLayers.csv"
    trend_filename = ( r"./re-organized_mls/") + trend_filename
    entropy_df.to_csv(trend_filename, index=False)

    # B. Automatically locate the "bottleneck layer" and export original Lag distribution (for use in the bottom graph of a 2x1 combination plot)
    # Find the layer closest to relative depth 0.5
    bottleneck_idx = (entropy_df['Relative_Depth'] - 0.5).abs().idxmin()
    target_layer_int = entropy_df.loc[bottleneck_idx, 'Layer_Int']

    # Extract all original Lag values for this layer from the significance dimension table
    raw_lags = significant_df[significant_df['Layer'] == target_layer_int][['Best_Lag_Sec']].copy()
    raw_lags['Model'] = MODEL_LABEL

    dist_filename = f"Raw_Lag_Dist_{MODEL_LABEL.replace('-', '_')}_moreLayers.csv"
    dist_filename = ( r"./re-organized_mls/") + dist_filename
    raw_lags.to_csv(dist_filename, index=False)

    print("-" * 30)
    print(f"[{MODEL_LABEL}] Data processing completed:")
    print(f"1. Trend data saved to: {trend_filename}")
    print(f"2. Bottleneck layer (L{target_layer_int}) distribution data saved to: {dist_filename}")
    print("-" * 30)

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()