"""
Please run in geometry_env:
geometry_env\Scripts\activate
"""

# env vars MUST come before all other imports
import os
import sys
from pathlib import Path

# Replace with the actual path to your FFmpeg bin folder
# ffmpeg_bin_path = r"D:\WPy64-31290\notebooks\ffmpeg-7.1.1-full_build-shared\bin"
# os.add_dll_directory(ffmpeg_bin_path)

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

# Preparing audios folder
AUDIO_PATH= script_dir / "audios"
os.makedirs(AUDIO_PATH, exist_ok=True)

# Huggingface Token
#os.environ["HF_TOKEN"] = "your_hugging_face_token_here"

# Mirror & Local cache
# os.environ["HF_ENDPOINT"]                     = "https://hf-mirror.com"
os.environ["HF_HOME"]                         = str(Path(f"{script_dir}/huggingface"))
os.environ["HF_DATASETS_CACHE"]               = str(Path(f"{script_dir}/huggingface/datasets"))
os.environ["HUGGINGFACE_HUB_CACHE"]           = str(Path(f"{script_dir}/huggingface/hub"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# other imports (after env vars)
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# Language dictionary
# key   : language name (used for directory/file naming)
# value : (fleurs_locale, description)
# All locales verified against google/fleurs dataset
# ─────────────────────────────────────────────────────────────────
LANGUAGE_DICT = {
    # tonal language gradient (test: tone count -> closed-loop topology)
    "vietnamese":  ("vi_vn",        "6 tones, Austroasiatic"),
    "cantonese":   ("yue_hant_hk",  "9 tones, Sino-Tibetan"),
    "thai":        ("th_th",        "5 tones, Kra-Dai"),
    "mandarin":    ("cmn_hans_cn",  "4 tones, Sino-Tibetan"),

    # rhythm/timing type (test: isochrony -> helix topology)
    "turkish":     ("tr_tr",        "vowel harmony, Turkic"),
    "finnish":     ("fi_fi",        "syllable-timed, Uralic"),
    "estonian":    ("et_ee",        "vowel length contrast, Uralic"),
    "arabic":      ("ar_eg",        "triconsonantal roots, Semitic"),

    # extreme phonological types (stress test)
    "georgian":    ("ka_ge",        "complex consonant clusters, Kartvelian"),
    "swahili":     ("sw_ke",        "CV syllables, Bantu"),
    "tamil":       ("ta_in",        "Dravidian, syllable-timed"),
    "zulu":        ("zu_za",        "click consonants, Bantu"),

    # genealogy vs phonological type comparison
    "dutch":       ("nl_nl",        "Germanic, between English and German"),
    "russian":     ("ru_ru",        "Slavic, complex consonant clusters"),
    "polish":      ("pl_pl",        "Slavic, very complex clusters"),
    "italian":     ("it_it",        "Romance, open syllables dominant"),
    "portuguese":  ("pt_br",        "Romance, nasal vowels"),
    "korean":      ("ko_kr",        "agglutinative, strict syllable structure"),
    "indonesian":  ("id_id",        "Austronesian, open syllables"),

    # other
    "spanish":     ("es_419",       "Romance, 5 vowels"),
    "french":      ("fr_fr",        "Romance, nasal vowels"),
    "german":      ("de_de",        "Germanic, consonant clusters"),
    "japanese":    ("ja_jp",        "mora-timed"),
    "english":     ("en_us",        "Germanic, schwa system"),
}

DATASET_ID = "google/fleurs"   # no token required, CC-BY-4.0

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
TARGET_DURATION   = 3600         # target duration in seconds (1 hour)
TARGET_SR         = 16000        # sample rate Hz (XLSR-53 requires 16kHz)
SHUFFLE_DATA      = True         # shuffle to mix speakers (recommended)
RANDOM_SEED       = 42
SHUFFLE_BUFFER    = 500          # streaming shuffle buffer size
DATASET_SPLIT     = "train"      # train split has ~10h per language

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def resample_audio(audio_array, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio_array
    try:
        import librosa
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        target_len = int(len(audio_array) / orig_sr * target_sr)
        return np.interp(
            np.linspace(0, len(audio_array) - 1, target_len),
            np.arange(len(audio_array)),
            audio_array
        )

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def download_and_merge(language_name):
    if language_name not in LANGUAGE_DICT:
        print(f"Error: {language_name!r} not in LANGUAGE_DICT.")
        print(f"Available: {', '.join(LANGUAGE_DICT.keys())}")
        return

    locale, description = LANGUAGE_DICT[language_name]

    output_dir  = os.path.join(script_dir, "audios", language_name+"_fleurs")
    output_file = os.path.join(output_dir, f"{language_name}_1_hours_merged.wav")
    # metadata (text transcription) saved next to the audio, in the same directory
    metadata_file = os.path.join(output_dir, f"{language_name}_raw_transcriptions.txt")
    os.makedirs(output_dir, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Language : {language_name} ({locale})")
    print(f"Info     : {description}")
    print(f"Dataset  : {DATASET_ID}  [no token required]")
    print(f"Mode     : streaming=True  (no full dataset cached locally)")
    print(f"Target   : {TARGET_DURATION}s  ({TARGET_DURATION/3600:.1f} hour)")
    print(f"Output   : {output_file}")
    print(f"Metadata : {metadata_file}  (raw transcriptions, one per segment)")
    print(f"{sep}\n")

    print("Connecting to HuggingFace FLEURS dataset (streaming)...")
    try:
        dataset = load_dataset(
            DATASET_ID,
            locale,
            split=DATASET_SPLIT,
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Check: 1) network/mirror  2) locale code in LANGUAGE_DICT")
        return

    print("Connected.")

    if SHUFFLE_DATA:
        print(f"Shuffling stream (buffer={SHUFFLE_BUFFER}, seed={RANDOM_SEED})...")
        dataset = dataset.shuffle(seed=RANDOM_SEED, buffer_size=SHUFFLE_BUFFER)

    print("Streaming and merging audio...\n")
    audio_segments = []
    # transcriptions are collected in the same order as the merged audio,
    # one entry per audio segment (line i matches the i-th segment of the wav)
    transcription_list = []
    total_duration = 0.0
    skipped = 0

    fmt = "{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]"
    with tqdm(total=TARGET_DURATION, unit="s", desc="Progress",
              bar_format=fmt) as pbar:
        for sample in dataset:
            if total_duration >= TARGET_DURATION:
                break
            try:
                arr     = np.array(sample["audio"]["array"], dtype=np.float32)
                orig_sr = sample["audio"]["sampling_rate"]
                if len(arr) == 0:
                    skipped += 1
                    continue
                if orig_sr != TARGET_SR:
                    arr = resample_audio(arr, orig_sr, TARGET_SR)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                dur = len(arr) / TARGET_SR
                rem = TARGET_DURATION - total_duration
                if dur > rem:
                    arr = arr[:int(rem * TARGET_SR)]
                    dur = rem
                audio_segments.append(arr)
                # capture the corresponding text transcription (metadata)
                transcription = sample.get("raw_transcription")
                if transcription is None:
                    transcription = sample.get("transcription", "")
                transcription_list.append(str(transcription).strip())
                total_duration += dur
                pbar.update(dur)
            except Exception:
                skipped += 1

    print()
    if total_duration < TARGET_DURATION * 0.9:
        print(f"Warning: only {total_duration:.1f}s collected "
              f"(< 90% of target). This language may lack sufficient data "
              f"in the train split.")
    else:
        print(f"Merge complete: {total_duration:.1f}s / "
              f"{total_duration/3600:.3f}h")

    if skipped:
        print(f"Skipped invalid segments: {skipped}")

    if not audio_segments:
        print("Error: no valid audio retrieved.")
        return

    print(f"\nWriting: {output_file}")
    merged = np.concatenate(audio_segments)
    sf.write(output_file, merged, TARGET_SR, subtype="PCM_16")

    # Write the metadata (raw transcriptions) to the same directory as the audio.
    # One line per segment, line i corresponds to the i-th segment of the wav.
    # A tab-separated format is used so it is easy to parse downstream.
    print(f"Writing: {metadata_file}")
    with open(metadata_file, "w", encoding="utf-8") as f:
        for i, trans in enumerate(transcription_list):
            start = sum(len(seg) for seg in audio_segments[:i]) / TARGET_SR
            end   = start + (len(audio_segments[i]) / TARGET_SR)
            f.write(f"{i}\t{start:.3f}\t{end:.3f}\t{trans}\n")

    mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Done!")
    print(f"  Duration : {len(merged)/TARGET_SR:.1f}s = "
          f"{len(merged)/TARGET_SR/3600:.3f}h")
    print(f"  Size     : {mb:.1f} MB")
    print(f"  Path     : {os.path.abspath(output_file)}")
    print(f"  Transcriptions : {len(transcription_list)} written to "
          f"{os.path.abspath(metadata_file)}")


if __name__ == "__main__":
    for SELECTED_LANGUAGE in LANGUAGE_DICT.keys():
        download_and_merge(SELECTED_LANGUAGE)
