import pandas as pd
import requests
import os
import numpy as np
import io
from pathlib import Path

# Get absolute path of this script
script_dir = Path(__file__).resolve().parent

print(f"Script Directory: {script_dir}")

OUTDIR=str(Path(f"{script_dir}/cldf-datasets_wals"))
os.makedirs(OUTDIR, exist_ok=True)

# --- 1. Configuration and Mapping Definitions ---
quant_maps = {
    'Syllable Structure': {1: 1, 2: 3, 3: 5}, 
    'Tone': {1: 0, 2: 1, 3: 2}, 
    'Consonant Inventories': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    'Vowel Quality Inventories': {1: 1, 2: 3, 3: 5}
}

def fetch_wals_matrix():
    base_url = "https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/"
    print("Fetching WALS data from GitHub...")
    languages = pd.read_csv(base_url + "languages.csv")[['ID', 'Name', 'ISO639P3code']].rename(columns={'Name': 'Language_Name'})
    values = pd.read_csv(base_url + "values.csv")[['Language_ID', 'Parameter_ID', 'Value']]
    parameters = pd.read_csv(base_url + "parameters.csv")[['ID', 'Name']].rename(columns={'Name': 'Feature_Name'})
    
    df_merged = pd.merge(values, parameters, left_on='Parameter_ID', right_on='ID')
    matrix_t = df_merged.pivot(index='Language_ID', columns='Feature_Name', values='Value')
    return pd.merge(languages, matrix_t, left_on='ID', right_index=True)

def fetch_phoible_features():
    """Aggregate using corrected PHOIBLE academic column names"""
    phoible_url = "https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv"
    print("Attempting to download PHOIBLE data (approx 30MB+)...")
    
    try:
        response = requests.get(phoible_url, timeout=120)
        response.raise_for_status()
        content = response.text.lstrip('\ufeff')
        pdf = pd.read_csv(io.StringIO(content), low_memory=False)
        
        # Unified column name cleaning
        pdf.columns = [str(c).strip().lower() for c in pdf.columns]
        
        # --- Mapping Logic Correction ---
        # Consonant column: 'consonantal'
        # Voiced column: 'periodicglottalsource' (physically defined voiced)
        # Phonemic unit: 'phoneme'
        
        required_cols = ['consonantal', 'periodicglottalsource', 'phoneme', 'iso6393']
        if not all(col in pdf.columns for col in required_cols):
            print(f"⚠ Detected column names do not match expectations. Column preview: {list(pdf.columns[:15])}")
            return None

        # 1. Filter consonants (consonantal == '+')
        consonants = pdf[pdf['consonantal'] == '+'].copy()
        
        # 2. Aggregate statistics
        # 'periodicglottalsource' being '+' indicates voiced/voiced source
        stats = consonants.groupby('iso6393').agg(
            total_consonants=('phoneme', 'count'),
            voiced_consonants=('periodicglottalsource', lambda x: (x == '+').sum())
        )
        
        # 3. Calculate voiced consonant ratio
        stats['voiced_ratio'] = stats['voiced_consonants'] / stats['total_consonants']
        
        print(f"PHOIBLE feature calculation successful! Valid languages: {len(stats)}")
        return stats[['voiced_ratio']].reset_index()

    except Exception as e:
        print(f"⚠ PHOIBLE processing failed: {e}")
        return None
        
# --- 2. Execution Logic ---
try:
    # --- WALS Processing ---
    cached_file_wals = os.path.join(OUTDIR, "wals_full_matrix.csv")
    if os.path.isfile(cached_file_wals):
        print("✓ Loading local WALS cache...")
        df_wals = pd.read_csv(cached_file_wals)
    else:
        df_wals = fetch_wals_matrix()
        df_wals.to_csv(cached_file_wals, index=False)

    # Extract quantified WALS key features
    selected_features = ['Language_Name', 'ISO639P3code', 'Syllable Structure', 'Tone', 'Consonant Inventories', 'Vowel Quality Inventories']
    existing_cols = [c for c in selected_features if c in df_wals.columns]
    T_matrix = df_wals[existing_cols].copy()

    for feature, mapping in quant_maps.items():
        if feature in T_matrix.columns:
            T_matrix[feature] = T_matrix[feature].map(mapping)

    # --- PHOIBLE Processing ---
    cached_file_phoible = os.path.join(OUTDIR, "phoible_processed.csv")
    if os.path.isfile(cached_file_phoible):
        print("✓ Loading local PHOIBLE aggregation cache...")
        phoible_df = pd.read_csv(cached_file_phoible)
    else:
        phoible_df = fetch_phoible_features()
        if phoible_df is not None:
            phoible_df.to_csv(cached_file_phoible, index=False)
        else:
            print("⚠ Failed to retrieve PHOIBLE data; feature library will lack 'Voiced_Ratio'.")

    # --- Merge ---
    if phoible_df is not None:
        print("Merging WALS and PHOIBLE by ISO code...")
        # Note: In fetch_phoible_features, we converted column names to lowercase
        T_matrix = pd.merge(T_matrix, phoible_df, left_on='ISO639P3code', right_on='iso6393', how='left')
        if 'iso6393' in T_matrix.columns:
            T_matrix = T_matrix.drop(columns=['iso6393'])
    else:
        T_matrix['voiced_ratio'] = np.nan

    # --- Final Cleaning and Normalization ---
    # Define numeric feature columns participating in normalization
    numeric_features = [c for c in T_matrix.columns if c not in ['Language_Name', 'ISO639P3code']]
    
    # Fill missing values with mean (Typological Average)
    T_matrix[numeric_features] = T_matrix[numeric_features].fillna(T_matrix[numeric_features].mean())
    # Fill extreme cases with 0 again
    T_matrix[numeric_features] = T_matrix[numeric_features].fillna(0)

    for col in numeric_features:
        max_v, min_v = T_matrix[col].max(), T_matrix[col].min()
        if max_v != min_v:
            T_matrix[col] = (T_matrix[col] - min_v) / (max_v - min_v)
        else:
            T_matrix[col] = 0.0

    print("\n>>> Final typological feature matrix construction complete!")
    print(T_matrix.head())

    # Save final results
    T_matrix.to_csv(os.path.join(OUTDIR, "typology_ground_truth.csv"), index=False)
    print(f"\nResults saved to {os.path.join(OUTDIR, "typology_ground_truth.csv")} (records: {len(T_matrix)})")

except Exception as e:
    print(f"\n[Fatal Error]: {e}")