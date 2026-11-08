import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# =========================================================
# 1. Automated Data Loading and Standardization (Added Normalization Step)
# =========================================================
def load_all_model_data():
    #path_2b = 'XLS_R_2B'
    #path_53 = 'XLSR_53'
    directories = [r"./re-organized_mls"]
    
    entropy_list = []
    lags_list = []
    
    # Define Dimension (Dimension D)
    D_MAP = {'XLS-R-2B': 1920, 'XLS-R-1B': 1280, 'XLSR-53': 1024, 'XLS-R-300M': 1024}
    
    for d in directories:
        e_files = glob.glob(os.path.join(d, "Entropy_Results_*.csv"))
        l_files = glob.glob(os.path.join(d, "Raw_Lag_Dist_*.csv"))
        
        for f in e_files:
            if not any(x in f for x in ["XLSR_53", "XLS_R_300M", "XLS_R_1B", "XLS_R_2B"]): continue
            temp_df = pd.read_csv(f)
            # --- Key Standardization Step: H_norm = H / log2(D) ---
            # We match D based on the filename or the Model column in the data
            for model_name, d_val in D_MAP.items():
                mask = temp_df['Model'].str.contains(model_name)
                if mask.any():
                    # Calculate normalized quotient
                    temp_df.loc[mask, 'H_norm'] = temp_df.loc[mask, 'Entropy'] / np.log2(d_val)
            
            entropy_list.append(temp_df)
            
        for f in l_files:
            if not any(x in f for x in ["XLSR_53", "XLS_R_300M", "XLS_R_1B", "XLS_R_2B"]): continue
            lags_list.append(pd.read_csv(f))

    if not entropy_list:
        raise FileNotFoundError("No data files found. Please check if exported CSVs exist in the specified path.")
        
    return pd.concat(entropy_list, ignore_index=True), pd.concat(lags_list, ignore_index=True)

# Execute loading
entropy_all, lags_all = load_all_model_data()

# Separate data for histogram plotting
lags_2b_data = lags_all[lags_all['Model'].str.contains('2B')]['Best_Lag_Sec']
lags_1b_data = lags_all[lags_all['Model'].str.contains('1B')]['Best_Lag_Sec']
lags_53_data = lags_all[lags_all['Model'].str.contains('53')]['Best_Lag_Sec']
lags_300_data = lags_all[lags_all['Model'].str.contains('300')]['Best_Lag_Sec']

# =========================================================
# 2. Plotting: Combined Canvas (Enhanced Annotations)
# =========================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13), gridspec_kw={'height_ratios': [1, 1]})
sns.set_style("whitegrid")

# ---------------------------------------------------------
# Top Plot: Normalized Entropy (H_norm)
# ---------------------------------------------------------
#palette = {'XLS-R-2B': '#1F77B4', 'XLS-R-1B': '#1F77B0', 'XLSR-53': '#FF7F0E', 'XLS-R-300M': '#007F0E'}
palette = {
    'XLS-R-2B':   '#E41A1C',   # bright red
    'XLS-R-1B':   '#377EB8',   # vivid blue
    'XLSR-53':    '#4DAF4A',   # green
    'XLS-R-300M': '#FF7F00',   # orange
}

sns.lineplot(
    data=entropy_all, 
    x='Relative_Depth', 
    y='H_norm', 
    hue='Model',
    style='Model', 
    palette=palette, 
    marker='s', 
    markersize=8, 
    linewidth=2.5, 
    ax=ax1
)

ax1.set_title(r'Normalized Temporal Entropy ($\mathcal{H}_{norm}$) across Model Scales', fontsize=15, fontweight='bold')
ax1.set_ylabel(r'Relative Capacity Utilization ($H / \log_2 D$)', fontsize=13)
ax1.set_xlabel('Relative Depth (Normalized)', fontsize=13)

# Automatically adjust Y-axis range to provide annotation space
y_min, y_max = entropy_all['H_norm'].min(), entropy_all['H_norm'].max()
ax1.set_ylim(y_min * 0.9, y_max * 1.15) 

ax1.axvline(0.5, color='red', linestyle=':', alpha=0.5)

# --- Smart Annotation Logic ---
for model_name, color in palette.items():
    model_subset = entropy_all[entropy_all['Model'].str.contains(model_name)]
    if not model_subset.empty:
        model_subset.loc[model_subset.index[: (len(model_subset['H_norm'])//2) ], 'H_norm'] = 0
        #print(model_subset)
        #input()

        #model_subset['H_norm'][:len(model_subset['H_norm'])//2] = [0] * (len(model_subset['H_norm']) // 2)
        # Find peak point
        peak_idx = model_subset['H_norm'].idxmax()
        peak_x = model_subset.loc[peak_idx, 'Relative_Depth']
        peak_y = model_subset.loc[peak_idx, 'H_norm']
        
        # Set different offset directions for 2B and 53 to prevent overlap
        if '2B' in model_name:
            offset_y = 0.05 * y_max
            text_pos = (peak_x + 0.05, peak_y + offset_y)
            label = f'2B Peak Utilization: {peak_y:.3f} (2.04 bits)'
        if '1B' in model_name:
            offset_y = 0.05 * y_max
            text_pos = (peak_x + 0.05, peak_y + offset_y)
            label = f'1B Peak Utilization: {peak_y:.3f} (2.03 bits)'
        elif '53' in model_name:
            offset_y = 0.05 * y_max
            text_pos = (peak_x - 0.5, peak_y + offset_y)
            label = f'53 Peak Utilization: {peak_y:.3f} (2.22 bits)'
        elif '300' in model_name:
            offset_y = 0.05 * y_max
            text_pos = (peak_x - 0.5, peak_y + offset_y)
            label = f'300M Peak Utilization: {peak_y:.3f} (2.97 bits)'
        #For bits value, consult: Entropy_Results_*.csv
        
        ax1.annotate(label, 
                     xy=(peak_x, peak_y), 
                     xytext=text_pos,
                     fontsize=11, fontweight='bold', color=color,
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle="arc3,rad=.2"))
# ---------------------------------------------------------
# Bottom Plot: Lag Distribution Comparison (Physical Verification)
# ---------------------------------------------------------
bin_width = 0.05
bins = np.arange(-1.5, 1.5 + bin_width, bin_width)

sns.histplot(lags_2b_data, color='#1F77B4', alpha=0.4, bins=bins, kde=True, 
             stat="probability", label='XLS-R-2B (Distributed / High H_norm)', ax=ax2)

sns.histplot(lags_1b_data, color='#1F77B0', alpha=0.4, bins=bins, kde=True, 
             stat="probability", label='XLS-R-11B (Distributed / High H_norm)', ax=ax2)

sns.histplot(lags_53_data, color='#FF7F0E', alpha=0.5, bins=bins, kde=True, 
             stat="probability", label='XLSR-53 (Crystallized / Low H_norm)', ax=ax2)

sns.histplot(lags_300_data, color='#007F0E', alpha=0.5, bins=bins, kde=True, 
             stat="probability", label='XLS-R-300 (Crystallized / Low H_norm)', ax=ax2)

ax2.set_title(r'Functional Lag Distribution at $\approx 0.5$ Relative Depth', fontsize=14)
ax2.set_xlabel('Time Lag (Best Lag Sec)', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.axvline(0, color='black', linestyle='--', alpha=0.3)

# Dynamic annotations
ax2.annotate('Crystallized / Greedy', xy=(-0.1, 0.25), xytext=(-1.1, 0.3), 
             fontsize=11, color='#FF7F0E', fontweight='bold',
             arrowprops=dict(facecolor='#FF7F0E', shrink=0.05, width=1.5))

ax2.annotate('Crystallized / Greedy', xy=(-0.2, 0.20), xytext=(-1.1, 0.25), 
             fontsize=11, color='#007F0E', fontweight='bold',
             arrowprops=dict(facecolor='#007F0E', shrink=0.05, width=1.5))
             
ax2.annotate('Parallel / Dispersed', xy=(0.2, 0.05), xytext=(-1.3, 0.15), 
             fontsize=11, color='#1F77B4', fontweight='bold',
             arrowprops=dict(facecolor='#1F77B4', shrink=0.05, width=1.5))
             
ax2.set_xlim(-1.5, 1.5)
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig(r"./re-organized_mls/Normalized_Temporal_Entropy_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# 3. Data Export: Save Standardized Results (Final Export)
# =========================================================

# Define export filename
output_csv = r"./re-organized_mls/Normalized_Temporal_Entropy_Results.csv"

# Export summarized data including H_norm column
try:
    # Keep only key columns to maintain file cleanliness
    export_cols = ['Model', 'Layer', 'Relative_Depth', 'Entropy', 'H_norm']
    # If original data has Language or other important dimensions, they can also be included
    existing_cols = [c for c in export_cols if c in entropy_all.columns]
    
    entropy_all[existing_cols].to_csv(output_csv, index=False)
    print(f"\n" + "="*50)
    print(f"Successfully exported standardized entropy data to: {output_csv}")
    print(f"Included models: {entropy_all['Model'].unique()}")
    print(f"2B Peak H_norm: {entropy_all[entropy_all['Model'].str.contains('2B')]['H_norm'].max():.4f}")
    print(f"1B Peak H_norm: {entropy_all[entropy_all['Model'].str.contains('1B')]['H_norm'].max():.4f}")
    print(f"53 Peak H_norm: {entropy_all[entropy_all['Model'].str.contains('53')]['H_norm'].max():.4f}")
    print(f"300 Peak H_norm: {entropy_all[entropy_all['Model'].str.contains('300')]['H_norm'].max():.4f}")
    print("="*50 + "\n")
    
except Exception as e:
    print(f"Export failed: {e}")

# If you need to save a subset separately for the 2B model
entropy_all[entropy_all['Model'].str.contains('2B')].to_csv(r"./re-organized_mls/Normalized_Temporal_Entropy_Results_XLS-R-2B_Only.csv", index=False)