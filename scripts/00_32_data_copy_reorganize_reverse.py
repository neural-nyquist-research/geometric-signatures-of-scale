import os
import sys
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

# Configuration information remains consistent
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
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser(description="Reverse operation: Move files back to inferred folder.")
    parser.add_argument("input", help="natural or shuffled?")
    parser.add_argument("model", help="model name?")
    args = parser.parse_args()

    inputstr = args.input.lower()

    # 1. Basic validation
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

    script_dir = Path(__file__).resolve().parent

    # 2. Replace .py with .log
    # Note: This assumes the filename ends with .py; add a check if unsure
    if script_name.endswith('.py'):
        log_filename = script_name[:-3] + suffix + '_' + modelname + '.log' # Remove last 3 characters (.py), add .log
    else:
        log_filename = script_name + suffix + '_' + modelname + '.log'

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


    # 3. Path definition (Reverse: move from DST back to SRC)
    # Current location (original DST)
    ORG_BASE = script_dir / f"re-organized{suffix}" / modelname
    # Target location (original SRC)
    INF_ROOT = script_dir / f"inferred{suffix}" / modelname

    if not ORG_BASE.exists():
        print(f"❌ Source directory not found: {ORG_BASE}")
        sys.exit(1)

    print(f"🔄 Reversing: {ORG_BASE} -> {INF_ROOT}")

    # 4. Core execution logic
    layers_count = configs[modelname]['layers']
    
    for layer_idx in range(layers_count):
        hs_folder_name = f"hidden_states_{layer_idx:02d}"
        layer_folder_name = f"hiddenlayer={layer_idx}"
        
        # Search for all language folders at this level under the current re-organized directory
        current_layer_path = ORG_BASE / hs_folder_name
        if not current_layer_path.exists():
            continue

        # Get all language folder names at this level (e.g., arabic_fleurs)
        lang_dirs = [d.name for d in current_layer_path.iterdir() if d.is_dir()]
        
        print(f"\n📦 Restoring {hs_folder_name}...")
        
        for lang_dir in tqdm(lang_dirs, desc="Restoring languages", leave=False):
            # File name
            file_name = "X_flat_stable_1hours.npy"
            
            # [Current Location]
            # re-organized/model/hidden_states_00/lang/hiddenlayer=0/file.npy
            src_file_path = ORG_BASE / hs_folder_name / lang_dir / layer_folder_name / file_name
            
            # [Target Location]
            # inferred/model/lang/hiddenlayer=0/file.npy
            dst_dir = INF_ROOT / lang_dir / layer_folder_name
            dst_file_path = dst_dir / file_name

            # Execute move
            if src_file_path.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # If target already exists, delete it first (to avoid naming conflicts)
                    if dst_file_path.exists():
                        os.remove(dst_file_path)
                    
                    shutil.move(str(src_file_path), str(dst_file_path))
                except Exception as e:
                    print(f"\n❌ Failed to move {lang_dir} layer {layer_idx}: {e}")

    print("\n" + "="*60)
    print("✅ All files have been moved back to the original 'inferred' structure!")
    print(f"Target directory: {INF_ROOT}")
    print("="*60)

if __name__ == "__main__":
    main()