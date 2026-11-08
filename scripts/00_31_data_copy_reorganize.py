import os
import sys
import shutil
from tqdm import tqdm
import shutil
import argparse

# Configuration
configs = {
    "wav2vec2-xls-r-2b" : {"layers": 49, "dim": 1920},
    "wav2vec2-large-xlsr-53" : {"layers": 25, "dim": 1024},
    "wav2vec2-xls-r-1b": {"layers": 49, "dim": 1280},
    "wav2vec2-xls-r-300m" : {"layers": 25, "dim": 1024},
    "seamless-m4t-v2-large" : {"layers": 25, "dim": 1024},
    "whisper-large-v2" : {"layers": 33, "dim": 1280},
    "whisper-large-v3" : {"layers": 33, "dim": 1280},
    "whisper-medium" : {"layers": 25, "dim": 1024},
    "mms-300m" : {"layers": 25, "dim": 1024},
    "mms-1b" : {"layers": 49, "dim": 1280},
    "mHuBERT-147" : {"layers": 13, "dim": 768}
}

def main():

    BIRDSONG_FILENAME = "birdsong_1hours_260320_0644_16K"

    # 1. Get the current script's filename (including .py)
    script_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="Please supply the following argument:"
    )
    parser.add_argument("input",  help="natural or shuffled?")
    parser.add_argument("model",  help="model name?")

    args = parser.parse_args()

    inputstr = args.input.lower()

    if inputstr not in ["natural", "shuffled"]:
        print(f"Usage: {script_name} [natural | shuffled] model")
        sys.exit(1)

    suffix = "_shuffled" if inputstr == "shuffled" else ""

    if args.model not in configs:
        print("Please give correct model name:")
        for name in configs:
            print(name)
        sys.exit(1)

    modelname = args.model

    # 2. Replace .py with .log
    # Note: This assumes the filename ends with .py; add a check if unsure
    if script_name.endswith('.py'):
        log_filename = script_name[:-3] + suffix + '_' + modelname + '.log' # Remove last 3 characters (.py), add .log
    else:
        log_filename = script_name + suffix + '_' + modelname  + '.log'

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

    from pathlib import Path

    # Get absolute path of this script
    script_dir = Path(__file__).resolve().parent

    print(f"Script Directory: {script_dir}")

    
    # ══════════════════════════════════════════════════════
    # Path Configuration (Please modify according to your actual drive letters)
    # ══════════════════════════════════════════════════════
    # SRC
    BASE_PATH = "inferred"+suffix
    SRC_ROOT = script_dir / BASE_PATH / modelname

    # DST
    DST_PATH = "re-organized"+suffix
    DST_BASE = script_dir / DST_PATH / modelname
    DST_BASE.mkdir(parents=True, exist_ok=True)
    
    # List of language folders
    LANG_FOLDERS = []
        
    for wav_path in SRC_ROOT.rglob("*.wav"):
        parent_name = wav_path.parent.name
        LANG_FOLDERS.append(parent_name)
 
    LANG_FOLDERS.sort()

    # ══════════════════════════════════════════════════════
    # Execution Logic
    # ══════════════════════════════════════════════════════

    # First loop: Create and populate hidden_states
    for layer_idx in range(configs[modelname]['layers']):
        hs_folder_name = f"hidden_states_{layer_idx:02d}"
        
        print(f"\n🚀 Generating directory structure: {hs_folder_name}")
        
        # Second loop: Process each language
        for lang_dir in tqdm(LANG_FOLDERS, desc=f"Processing languages", leave=False):
            
            # --- Determine source file path ---
            src_sub_dir = f"hiddenlayer={layer_idx}"
            src_file_name = "X_flat_stable_1hours.npy"
            
            src_path = os.path.join(SRC_ROOT, lang_dir, src_sub_dir, src_file_name)
            
            # --- Determine target file path ---
            # Structure example: ...\hidden_states_00\arabic_fleurs\hiddenlayer=0\X_flat_stable_1hours.npy
            dst_folder_name = f"hiddenlayer={layer_idx}"
            dst_dir = os.path.join(DST_BASE, hs_folder_name, lang_dir, dst_folder_name)
            dst_path = os.path.join(dst_dir, src_file_name)
            
            # --- Execute copy operation ---
            if os.path.exists(src_path):
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                
                # Skip if target file already exists to speed up multiple runs
                if not os.path.exists(dst_path):
                    try:
                        #shutil.copy2(src_path, dst_path)
                        shutil.move(src_path, dst_path)
                    except Exception as e:
                        print(f"\n❌ Error: Unable to copy layer {layer_idx} ({lang_dir}) -> {e}")
            else:
                # Only warn about missing source files during the hidden_states_00 loop to avoid log spamming
                if layer_idx == 0:
                    print(f"\n⚠️ Missing source file: {src_path}")

    print("\n" + "="*60)
    print("✅ Directory reorganization for all hidden_states is complete!")
    print(f"Check main directory: {DST_BASE}")
    print("="*60)
    
    # Close log file and restore stdout
    sys.stdout.close()
    
if __name__ == "__main__":
    main()