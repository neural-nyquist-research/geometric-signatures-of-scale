import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

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

# --- 1. Load Data ---
neuron_1815 = np.load("_hn_rucheng_dim1815_frames1000_2000.npy")
praat_f0 = np.load("praat_f0_aligned_20s_40s.npy")

# --- 2. Normalize (Z-score for direct amplitude comparison) ---
def robust_zscore(arr):
    mu = np.nanmean(arr)
    std = np.nanstd(arr)
    return (arr - mu) / std

neuron_z = robust_zscore(neuron_1815)
f0_z = robust_zscore(praat_f0)

# --- 3. APPLY THE 1980MS SHIFT ---
lag_ms = 1980
lag_frames = int(lag_ms / 20)  # = 99 frames

# IMPORTANT: We shift the NEURAL data backward 
# to align it with the ACOUSTIC event that caused it.
neuron_shifted = np.roll(neuron_z, -lag_frames)

# Because np.roll wraps around, we need to blank the last 99 frames
# where the data is now invalid.
neuron_shifted[-lag_frames:] = np.nan

# --- 4. Plotting the Phase-Locked Overlay ---
fig, ax = plt.subplots(figsize=(16, 6))

# Main synchronized plot
# We shorten the plot window slightly to exclude the invalid rolled-over data
plot_window = len(f0_z) - lag_frames

ax.plot(f0_z[:plot_window], color='dodgerblue', label='Praat $F_0$ (Ground Truth)', alpha=0.5, lw=3, linestyle='--')
ax.plot(neuron_shifted[:plot_window], color='crimson', label=f'Tone Expert (Shifted -{lag_ms}ms)', alpha=0.9, lw=2)

# --- THE KEY DIAGNOSTIC OVERLAYS ---

# 1. Unvoiced Mask (Proof the neuron only "hears" F0)
unvoiced_mask = np.isnan(praat_f0[:plot_window])
ax.fill_between(np.arange(plot_window), -3, 3, where=unvoiced_mask, color='gray', alpha=0.15, label='Unvoiced Zone')

# 2. Add Phase-Lock markers to the first major peak
# Find the first peak of F0 to highlight the synchronization
f0_subset = f0_z[150:350]
if np.sum(~np.isnan(f0_subset)) > 0:
    first_peak_f0 = np.nanargmax(f0_subset) + 150
    # Add a visual "consonance marker" (a vertical dashed line linking peaks)
    ax.axvline(x=first_peak_f0, color='black', linestyle=':', alpha=0.5, ymin=0.3, ymax=0.9)
    ax.text(first_peak_f0 - 5, 2.5, 'Phase Locked', color='black', fontsize=10, fontweight='bold', horizontalalignment='right')

# Styling
ax.set_title(f"Phase-Locked Synchronization (Δt = {lag_ms}ms Shift Applied)", fontsize=18, fontweight='bold')
ax.set_ylabel("Standardized Amplitude (Z-score)", fontsize=14)
ax.set_xlabel("Time (Frames, 20ms/frame)", fontsize=14)
ax.set_ylim(-2.5, 3.0) # Standardize Y-axis for publication
ax.legend(loc='upper right', frameon=True, fontsize=11, facecolor='white')
ax.grid(True, linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig(f"praat_sync_shifted_{lag_ms}ms.png", dpi=300)
plt.show()

print(f">>> Figure generated: praat_sync_shifted_{lag_ms}ms.png")

print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Close log file and restore stdout
if hasattr(sys.stdout, 'close'):
    sys.stdout.close()
