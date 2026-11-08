def main():

    import sys
    import os
    import gc
    import numpy as np
    import torch
    import torchaudio
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from tqdm import tqdm
    import math
    import shutil
    import argparse
    from datetime import datetime
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = "facebook/wav2vec2-xls-r-2b"
    modelname = MODEL_NAME.split("/")[1]

    # 1. Get the current script's filename (including .py)
    script_name = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="Please supply the following argument:"
    )
    parser.add_argument("input",  help="natural or shuffled?")

    args = parser.parse_args()

    inputstr = args.input
    print(inputstr)
    if inputstr.lower() != "natural" and inputstr.lower() != "shuffled" and inputstr.lower() != "mls":
        print(f"Usage: {script_name} [natural | mls | shuffled]")
        sys.exit(1)

    if inputstr.lower() == "shuffled":
        suffix= "_shuffled"
    elif inputstr.lower() == "mls":
        suffix_folder= "_mls_data"
        suffix=""
    else:
        suffix=""

    # 2. Replace .py with .log
    # Note: This assumes the filename ends with .py; add a check if unsure
    if script_name.endswith('.py'):
        log_filename = script_name[:-3] + suffix_folder + '.log' # Remove last 3 characters (.py), add .log
    else:
        log_filename = script_name + suffix_folder + '.log'

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

    from pathlib import Path

    # Get absolute path of this script
    script_dir = Path(__file__).resolve().parent

    print(f"Script Directory: {script_dir}")

    # Construct a path to the ffmpeg/bin folder inside the same folder as the script
    FFMPEG_PATHHERE = script_dir / "ffmpeg-7.1.1-full_build-shared/bin"
    print(f"ffmpeg Path: {FFMPEG_PATHHERE}")

    # ============================================
    # FFmpeg path detection and DLL environment configuration
    # ============================================
    # ============================================
    # Cross-Platform FFmpeg Detection
    # ============================================
    def find_and_setup_ffmpeg():
        import shutil
            
        # --- LINUX LOGIC ---
        if sys.platform != 'win32':
            ffmpeg_exe = shutil.which('ffmpeg')
            if ffmpeg_exe:
                print(f"Linux detected. Using system FFmpeg: {ffmpeg_exe}")
                return os.path.dirname(ffmpeg_exe)
            raise Exception("FFmpeg not found in Linux PATH. Did you install it in the Dockerfile?")

        # --- WINDOWS LOGIC ---
        """
        Find ffmpeg and explicitly set Windows DLL search directory.
        No longer relies on PATH environment variable, but uses os.add_dll_directory instead.
        """
        
        # 1. Try to get from environment variable (if user has already set it)
        ffmpeg_path = os.environ.get('FFMPEG_PATH', FFMPEG_PATHHERE)
        if ffmpeg_path and os.path.exists(os.path.join(ffmpeg_path, 'ffmpeg.exe')):
            return ffmpeg_path

        # 2. Try common installation paths (optimized for your environment)
        common_paths = [
            Path(f"{script_dir}\ffmpeg-7.1.1-full_build-shared\bin"),
            r"C:\facebook_multilingual_librispeech\ffmpeg-7.1.1-full_build-shared\bin",
            r"D:\facebook_multilingual_librispeech\ffmpeg-7.1.1-full_build-shared\bin",
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
        ]
        
        for path in common_paths:
            if os.path.exists(os.path.join(path, 'ffmpeg.exe')):
                print(f"Found FFmpeg at preset path: {path}")
                return path
        
        # 3. Find ffmpeg in system PATH using shutil.which
        ffmpeg_exe = shutil.which('ffmpeg')
        if ffmpeg_exe:
            bin_dir = os.path.dirname(ffmpeg_exe)
            print(f"Found FFmpeg via system PATH: {bin_dir}")
            return bin_dir
        
        raise Exception("FFmpeg not found. Please ensure the 'full_build-shared' version is installed, or add its bin directory to the system PATH.")

    # Execute search and get bin directory
    ffmpeg_bin = find_and_setup_ffmpeg()
    print(f"Confirmed FFmpeg directory usage: {ffmpeg_bin}")

    # ============================================
    # Windows DLL loading configuration (critical step)
    # ============================================
    import sys
    if sys.platform == 'win32':
        try:
            # [Core Fix] Explicitly add DLL search directory
            # This will make Python automatically look for .dll files in ffmpeg_bin when loading torchcodec and its dependencies
            os.add_dll_directory(ffmpeg_bin)
            
            # Optional: If add_dll_directory fails (older Windows), fallback to modifying PATH (though usually not needed on Win10/11)
            if hasattr(os, 'add_dll_directory'):
                print(f"Explicitly added DLL search directory: {ffmpeg_bin}")
                
                # [Optional Optimization] If fonts.conf is also needed, keep this logic, but no need to pollute PATH anymore
                fonts_conf_path = os.path.join(ffmpeg_bin, "fonts.conf")
                if not os.path.exists(fonts_conf_path):
                    print(f"Creating fonts.conf file: {fonts_conf_path}")
                    with open(fonts_conf_path, 'w', encoding='utf-8') as f:
                        f.write('''<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>C:/Windows/Fonts</dir>
  <cachedir>./fontconfig_cache</cachedir>
  <config><rescan><int>30</int></rescan></config>
</fontconfig>''')
                
                # [Important] Set FONTCONFIG environment variable for internal ffmpeg use, but no longer concatenate PATH
                os.environ['FONTCONFIG_PATH'] = str(ffmpeg_bin)
                os.environ['FONTCONFIG_FILE'] = 'fonts.conf'
                
        except Exception as e:
            print(f"Warning: Unable to add DLL directory (possibly older Windows), attempting fallback to PATH modification...")
            # If add_dll_directory is unavailable, use modifying PATH as the last resort
            os.environ['PATH'] = str(ffmpeg_bin) + os.pathsep + os.environ.get('PATH', '')

    # Set offline mode environment variables
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"FFmpeg environment configuration complete. About to import torchcodec...")

    #import torchcodec

    MODEL_PATH = str(script_dir / "models")

    BASE_DIR = str(script_dir / "inferred")
    BASE_DIR += suffix
    os.makedirs(BASE_DIR, exist_ok=True)
    BASE_DIR+=("/"+modelname)
    os.makedirs(BASE_DIR, exist_ok=True)

    audio_dir = Path(f"{script_dir}/audios{suffix_folder}")
    output_base = Path(BASE_DIR)

    # List of language folders
    LANG_FOLDERS = []
        
    for wav_path in audio_dir.rglob("*.wav"):
            
        parent_name = wav_path.parent.name
        
        LANG_FOLDERS.append(parent_name)
        
        target_dir = output_base / parent_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file path
        output_path = str(target_dir / wav_path.name)
        wav_path_str = str(wav_path)
        
        if not os.path.exists(output_path):
            try:
                shutil.copy2(wav_path_str, output_path)
                print(f"✅ Wave file copied: {wav_path_str}")
            except Exception as e:
                print(f"\n❌ Error: Cannot copy {wav_path_str}")
        else:
            print(f"✅ Wave file already existed: {wav_path_str}")

    LANG_FOLDERS.sort()

    # ╔══════════════════════════════════════════════════════╗
    # ║        ★ User Configuration Area (Localized) ★       ║
    # ╚══════════════════════════════════════════════════════╝

    # ──────────────────────────────────────────────────────
    # Local model cache path (leave as empty string "" to auto-download from HuggingFace Hub)
    # LOCAL_MODEL_PATH = ""
    LOCAL_MODEL_PATH = MODEL_PATH + "/" + modelname

    FRAME_RATE  = 50
    WINDOW_SIZE = 10
    STRIDE      = WINDOW_SIZE // 2   # = 5, automatic follow-up
    CHUNK_SEC   = 10                 # Process 10 seconds at a time for memory efficiency
    NUM_LAYERS  = 49                 # Layers 0 to 24, totaling 49 layers
    # ──────────────────────────────────────────────────────

    to_do        = list(range(49))  # Process all 49 layers; can also be changed to [2,5,8,...] to process only specific layers
    #to_do = [19]
    BATCH_LAYERS = 49               # ← Number of layers processed per batch, adjust based on memory:
                                    #    4 layers ≈ Peak 8GB (conservative)
                                    #    8 layers ≈ Peak 16GB
                                    #   13 layers ≈ Peak 32GB
                                    #   49 layers ≈ Peak 96GB (requires sufficient memory)

    def _resolve_model_path(local: str, hub: str) -> str:
        """Prioritizes local path; otherwise uses the Hub ID."""
        if local and os.path.isdir(local):
            print(f"[Model Path] Using local cache: {local}")
            return local
        print(f"[Model Path] Local path not configured or does not exist, using Hub ID: {hub}")
        return hub

    MODEL_PATH     = _resolve_model_path(LOCAL_MODEL_PATH, MODEL_NAME)

    # Load model
    print("Loading model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2Model.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    print("Model loading complete")

    # ══════════════════════════════════════════════════════
    # Loop to process each language
    # ══════════════════════════════════════════════════════
    for lang_dir in LANG_FOLDERS:
        # Extract language name
        if "mls_spanish" in lang_dir: # mls_spanish
            lang_name = lang_dir.split('_')[1]
        elif "fleurs" in lang_dir or "mls" in lang_dir: # e.g., extract arabic from arabic_fleurs
            lang_name = lang_dir.split('_')[0]
        elif "rucheng" in lang_dir: #_hn_rucheng
            lang_name = lang_dir[1:]
        else:
            print("Language name error!")
            exit(1)
            
        audio_file = os.path.join(BASE_DIR, lang_dir, f"{lang_name}_1_hours_merged{suffix}.wav")
        
        if not os.path.exists(audio_file):
            print(f"⚠️ Skipping: Audio file {audio_file} not found")
            continue

        print(f"\n" + "="*70)
        print(f"🌍 Processing language: {lang_name.upper()}")
        print(f"Audio path: {audio_file}")
        print("="*70)

        # Load audio
        waveform, sr = torchaudio.load(audio_file)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform_np = waveform.squeeze().numpy()
        total_sec   = len(waveform_np) / sr
        print(f"Audio duration: {total_sec:.1f}s")

        # Chunking parameters
        chunk_samples   = CHUNK_SEC * sr
        overlap_sec     = 0.5
        overlap_samples = int(overlap_sec * sr)
        starts          = range(0, len(waveform_np), chunk_samples)

        # Split to_do into batches, with BATCH_LAYERS layers per batch
        batches = [to_do[i:i + BATCH_LAYERS] for i in range(0, len(to_do), BATCH_LAYERS)]
        print(f"Total {len(starts)} segments, each {CHUNK_SEC}s")
        print(f"to_do contains {len(to_do)} layers, split into {len(batches)} batches, max {BATCH_LAYERS} layers per batch")
        print(f"Layer numbers in each batch: {batches}")

        # ══════════════════════════════════════════════════════
        # Outer Loop: Batch Iteration
        # ══════════════════════════════════════════════════════
        for batch_idx, batch_layers in enumerate(batches):
            print(f"\n{'═'*60}")
            print(f"  >>> Batch {batch_idx+1}/{len(batches)}, processing layers: {batch_layers}")
            print(f"{'═'*60}")

            # Allocate memory only for the layers in this batch
            all_features_per_layer = {layer_idx: [] for layer_idx in batch_layers}

            for i, start in enumerate(tqdm(starts, desc=f"Segment extraction batch{batch_idx+1}/{len(batches)}")):
                end   = min(start + chunk_samples + overlap_samples, len(waveform_np))
                chunk = waveform_np[start:end]

                inputs       = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
                input_values = inputs["input_values"]

                with torch.no_grad():
                    outputs       = model(input_values, output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states  # tuple, length NUM_LAYERS

                # Save only the layers in this batch
                for layer_idx in batch_layers:
                    features = hidden_states[layer_idx].squeeze(0).cpu().numpy()
                    if i < len(starts) - 1:
                        keep_frames = int(CHUNK_SEC * FRAME_RATE)
                        features    = features[:keep_frames]
                    all_features_per_layer[layer_idx].append(features)

            print(f"Batch {batch_idx+1} segment extraction complete, starting layer-by-layer processing...")

            # ══════════════════════════════════════════════════════
            # Inner Loop: Layer-by-Layer Processing (for each layer in the current batch)
            # ══════════════════════════════════════════════════════
            for layer_idx in batch_layers:

                print(f"\n{'#'*60}")
                print(f"# Current loop index: Layer {layer_idx} (hiddenlayer={layer_idx})")
                print(f"{'#'*60}")

                # ──────────────────────────────────────────────────
                # Concatenate, calculate stable frames, save
                # ──────────────────────────────────────────────────
                print(f"\nConcatenating features, calculating stable frames and saving (layer={layer_idx})")

                features_all = np.concatenate(all_features_per_layer[layer_idx], axis=0)
                print(f"Complete features shape: {features_all.shape}")

                clustered_features = []
                for i in range(0, len(features_all) - WINDOW_SIZE, STRIDE):
                    window     = features_all[i : i + WINDOW_SIZE]
                    mean_vec   = np.mean(window, axis=0)
                    dists      = np.linalg.norm(window - mean_vec, axis=1)
                    stable_idx = np.argmin(dists)
                    clustered_features.append(window[stable_idx])

                X_flat_stable = np.array(clustered_features)
                print(f"Number of feature vectors: {len(X_flat_stable)}  Dimension: {X_flat_stable.shape[1]}")

                # Create directory and save
                
                save_folder = os.path.join(BASE_DIR, lang_dir, f"hiddenlayer={layer_idx}")
                os.makedirs(save_folder, exist_ok=True)
                
                # Saving stable frames
                np.save(os.path.join(save_folder, 'X_flat_stable_1hours.npy'), X_flat_stable)
                # If you want to save all frames for hidden layer activations
                # np.save(os.path.join(save_folder, 'features_all_1hours.npy'), features_all)
                
                # Free memory
                del features_all, X_flat_stable
                del all_features_per_layer[layer_idx]   # Release VRAM/memory promptly


            del all_features_per_layer
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "★"*60)
    print("  All language processing tasks completed successfully!")
    print("★"*60)

    print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Close log file and restore stdout
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    
if __name__ == "__main__":
    main()
