import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial import ConvexHull

# =========================================================
# 1. Plotting Style and Parameter Settings
# =========================================================
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
point_size = 60
alpha_val = 0.6
hull_color = '#d62728' # Prominent red, representing physical boundaries

# =========================================================
# 2. Utility Function: Calculate and Plot Convex Hull
# =========================================================
def plot_manifold_dispersion(ax, points, title, color_scatter, area_val):
    """
    Plot point cloud and convex hull on the given ax, with area annotation
    """
    if len(points) < 3:
        ax.text(0.5, 0.5, "Insufficient Points", ha='center')
        return

    # Plot real point cloud
    ax.scatter(points[:, 0], points[:, 1], color=color_scatter, s=point_size, 
               alpha=alpha_val, edgecolors='none', label='Phonetic Anchors')
    
    # Calculate and plot Convex Hull
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], color=hull_color, linewidth=2.0)
    
    # Fill convex hull area with red shadow to enhance visual contrast
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=hull_color, alpha=0.15)
    
    # Decoration
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Max Correlation ($r_{max}$)', fontsize=13)
    ax.set_ylabel('Best Lag Sec ($\tau_{best}$)', fontsize=13)
    
    # Annotate core data (area)
    # ax.text(-0.9, -0.6, f'Coverage Area: {area_val:.4f}', fontsize=12, 
    #         fontweight='bold', color=hull_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Set unified coordinate axis ranges for easy comparison
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(axis='both', alpha=0.2)
    ax.legend(loc='upper right', fontsize=10)

# =========================================================
# 3. Prepare Real Data (Please ensure paths are correct)
# =========================================================
# ==========================================
# Step One: Get significant_df_53
# ==========================================
# Please ensure the path points to your XLSR-53 experimental results file
csv_path_53 = r'./re-organized_mls/wav2vec2-large-xlsr-53/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv'

# Read and filter: keep only significant dimensions with absolute correlation > 0.1
df_53_raw = pd.read_csv(csv_path_53)
significant_df_53 = df_53_raw[df_53_raw['Max_Corr'].abs() > 0.1].copy()

print(f"Successfully loaded XLSR-53 data, totaling {len(significant_df_53)} significant dimensions.")

# Please ensure the path points to your XLSR-300M experimental results file
csv_path_300 = r'./re-organized_mls/wav2vec2-xls-r-300m/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv'

# Read and filter: keep only significant dimensions with absolute correlation > 0.1
df_300_raw = pd.read_csv(csv_path_300)
significant_df_300 = df_300_raw[df_300_raw['Max_Corr'].abs() > 0.1].copy()

print(f"Successfully loaded XLSR-300 data, totaling {len(significant_df_300)} significant dimensions.")

# Modify the path to point to the 2B results file
csv_path_2b = r'./re-organized_mls/wav2vec2-xls-r-2b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv'

# Use the same filtering threshold (|Max_Corr| > 0.1)
df_2b_raw = pd.read_csv(csv_path_2b)
significant_df_2b = df_2b_raw[df_2b_raw['Max_Corr'].abs() > 0.1].copy()
print(f"Successfully loaded XLS-R-2B data, totaling {len(significant_df_2b)} significant dimensions.")

# Modify the path to point to the 1B results file
csv_path_1b = r'./re-organized_mls/wav2vec2-xls-r-1b/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv'

# Use the same filtering threshold (|Max_Corr| > 0.1)
df_1b_raw = pd.read_csv(csv_path_1b)
significant_df_1b = df_1b_raw[df_1b_raw['Max_Corr'].abs() > 0.1].copy()
print(f"Successfully loaded XLS-R-1B data, totaling {len(significant_df_1b)} significant dimensions.")

# For area and norm, consult: F:\WPy64-31290_art01_rev\notebooks\re-organized_mls\manifold_area.csv

# Get point cloud for XLSR-53 L12
pts_53_l12 = significant_df_53[significant_df_53['Layer'] == 13][['Max_Corr', 'Best_Lag_Sec']].values
#area_53 = 0.2166 # Previously calculated value
#norm_53 = 0.0341
area_53 = 0.0268 # Previously calculated value
norm_53 = 0.0055

# Get point cloud for XLSR-300 L12
pts_300_l12 = significant_df_300[significant_df_300['Layer'] == 13][['Max_Corr', 'Best_Lag_Sec']].values
#area_300 = 0.1445 # Previously calculated value
#norm_300 = 0.0280
area_300 = 0.0039 # Previously calculated value
norm_300 = 0.0011

# Get point cloud for XLS-R-2B L25
pts_2b_l25 = significant_df_2b[significant_df_2b['Layer'] == 25][['Max_Corr', 'Best_Lag_Sec']].values
area_2b = 0.0212 # Previously calculated value
norm_2b = 0.0040

# Get point cloud for XLS-R-2B L25
pts_1b_l25 = significant_df_1b[significant_df_1b['Layer'] == 25][['Max_Corr', 'Best_Lag_Sec']].values
area_1b = 0.0155 # Previously calculated value
norm_1b = 0.0040

# formant_dim_analysis_06_corr-loop_02_moreLayers_normalized.txt

# =========================================================
# 4. Execute Plotting
# =========================================================
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

#For nodes, consult: F:\WPy64-31290_art01_rev\notebooks\32_03_formant_dim_analysis_06_corr-loop_01_moreLayers.log
#For mean shift, consult: F:\WPy64-31290_art01_rev\notebooks\32_05_formant_dim_analysis_06_corr-loop_03_meanshift_moreLayers.log

# Left plot: XLSR-53 L12 (Narrowband Manifold)
plot_manifold_dispersion(ax1, pts_53_l12, 'XLSR-53 (Layer 13) Bottleneck\nN=30 nodes', '#3182bd', area_53)
# Annotate corresponding Mean Shift
ax1.text(0.05, 0.8, f'Mean Shift = 112.26 ms\nCoverage Area (Normalized) = {norm_53:.4f} ({area_53:.4f})', 
         transform=ax1.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

# Middle plot1: XLSR-300 L12 (Narrowband Manifold)
plot_manifold_dispersion(ax2, pts_300_l12, 'XLS-R-300M (Layer 13) Bottleneck\nN=12 nodes', '#3182bd', area_300)
# Annotate corresponding Mean Shift
ax2.text(0.05, 0.8, f'Mean Shift = 39.85 ms\nCoverage Area (Normalized) = {norm_300:.4f} ({area_300:.4f})', 
         transform=ax2.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

# Middle plot2: XLS-R-1B L25 (Wide-area Nebula)
plot_manifold_dispersion(ax3, pts_1b_l25, 'XLS-R-1B (Layer 25) Bottleneck\nN=16 nodes', '#FF7F0E', area_1b)
ax3.text(0.05, 0.8, f'Mean Shift = 96.22 ms\nCoverage Area (Normalized) = {norm_1b:.4f} ({area_1b:.4f})', 
         transform=ax3.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
# Right plot: XLS-R-2B L25 (Wide-area Nebula)
plot_manifold_dispersion(ax4, pts_2b_l25, 'XLS-R-2B (Layer 25) Bottleneck\nN=41 nodes', '#FF7F0E', area_2b)
ax4.text(0.05, 0.8, f'Mean Shift = 28.68 ms\nCoverage Area (Normalized) = {norm_2b:.4f} ({area_2b:.4f})', 
         transform=ax4.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
         
# Decorate total figure
plt.suptitle('Manifold Expansion as a Geometric Driving Force of Scaling Laws', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(r"./re-organized_mls/Topological_scaling_real_data_moreLayers_L13.png", dpi=300, bbox_inches='tight')
plt.show()