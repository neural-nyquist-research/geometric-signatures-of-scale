import pandas as pd
import os
from pathlib import Path

# Get absolute path of this script
script_dir = Path(__file__).resolve().parent

print(f"Script Directory: {script_dir}")

OUTDIR=str(Path(f"{script_dir}/cldf-datasets_wals"))

def extract_target_languages(input_file, output_file):
    # 1. Load the complete Ground Truth data
    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    df = pd.read_csv(input_file)
    
    if df.empty:
        print("Error: Input file is empty.")
        return

    # 2. Define target languages and their preferred variants
    preferred_variants = {
        'Arabic': 'Arabic (Modern Standard)',  
        'Cantonese': 'Cantonese',                     
        'Dutch': 'Dutch',                      
        'English': 'English',                       
        'Estonian': 'Estonian',
        'Finnish': 'Finnish',
        'French': 'French',
        'Georgian': 'Georgian',
        'German': 'German (Berlin)',
        'Indonesian': 'Indonesian (Jakarta)',  
        'Italian': 'Italian',
        'Japanese': 'Japanese',
        'Korean': 'Korean',
        'Mandarin': 'Mandarin',
        'Polish': 'Polish',
        'Portuguese': 'Portuguese',
        'Russian': 'Russian',                  
        'Spanish': 'Spanish',
        'Swahili': 'Swahili',
        'Tamil': 'Tamil',
        'Thai': 'Thai',
        'Turkish': 'Turkish',
        'Vietnamese': 'Vietnamese',
        'Zulu': 'Zulu (Northern)',
    }

    extracted_rows = []
    
    # 3. Extraction logic (Exact match on the preferred variant string)
    for target_lang in preferred_variants.keys():
        preferred_variant = preferred_variants[target_lang]
        
        # Find rows that exactly match the specific variant we want to extract
        mask = df['Language_Name'].str.strip() == preferred_variant
        match = df[mask]
        
        if not match.empty:
            # Take the first matching row
            extracted_rows.append(match.iloc[0].to_frame().T)
        else:
            print(f"⚠ Language '{target_lang}' not found in the database")

    if extracted_rows:
        # Merge the extracted rows
        subset_df = pd.concat(extracted_rows, ignore_index=True)
        
        # Remove duplicates based on ISO code (keep first occurrence)
        subset_df = subset_df.drop_duplicates(subset=['ISO639P3code'], keep='first')

        # --- SIMPLIFIED RENAMING LOGIC ---
        # Create a reverse mapping: Variant Name -> Target Key
        variant_to_key = {v: k for k, v in preferred_variants.items()}
        
        # Rename the Language_Name column to the dictionary key
        subset_df['Language_Name'] = subset_df['Language_Name'].apply(
            lambda x: variant_to_key.get(x.strip(), x)
        )
        # --- End of Renaming Logic ---

        # Sort according to the order in target_names list
        target_names_list = list(preferred_variants.keys())
        
        # Now that Language_Name is already the key, we can just sort by index directly
        subset_df['Sort_Order'] = subset_df['Language_Name'].apply(
            lambda x: target_names_list.index(x) if x in target_names_list else 999
        )
        
        subset_df = subset_df.sort_values('Sort_Order').drop(columns=['Sort_Order'])
        
        # --- NEW LOGIC: Append the specific row before saving ---
        # We create a temporary DataFrame for this single row to match the structure of subset_df
        # Assuming your CSV columns are at least: Language_Name, ISO639P3code, plus the extra fields
        # If your CSV has many columns, we need to ensure the new row aligns with them.
        
        # Let's assume the output file should have these specific columns if they aren't already in df.
        # However, since you are appending a row, we need to handle potential NaNs for other columns.
        
        # Create the new row as a Series
        new_row_data = {
            'Language_Name': '_hn_rucheng',
            'ISO639P3code': 'hak',
            'Syllable Structure': '0.5',
            'Tone': '1.0',
            'Consonant Inventories': '0.25',
            'Vowel Quality Inventories': '0.5',
            'voiced_ratio': '0.3287671232876712'
        }
        
        # If there are other columns in subset_df (e.g., if your original CSV had more data),
        # we need to add them as NaN or default values. 
        # Based on the string provided: _hn_rucheng,hak,0.5,1.0,0.25,0.5,0.328...
        # It seems you might want these specific columns added if they don't exist, 
        # OR just appended to the existing rows.
        
        # Let's assume the CSV structure is flexible or we are creating a minimal view.
        # To be safe, let's create a new DataFrame for this row and fill missing columns with NaN
        
        new_row_df = pd.DataFrame([new_row_data])
        
        # Align columns: If subset_df has more columns than the new row, 
        # pandas concat will handle it by filling missing values with NaN.
        
        subset_df = pd.concat([subset_df, new_row_df], ignore_index=True)
        
        # Optional: If you want to move this new row to a specific position (e.g., first or last),
        # you can sort again or reorder. Here we just append it.
        
        # --- End of Append Logic ---

        # Export
        subset_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n>>> Successfully extracted data for {len(subset_df)} target languages to: {output_file}")
        print(subset_df[['Language_Name', 'ISO639P3code']])
    else:
        print("No data was extracted.")

# --- Execute extraction ---


extract_target_languages(os.path.join(OUTDIR, "typology_ground_truth.csv"),
                         os.path.join(OUTDIR, "target_languages_typology.csv")
                         )