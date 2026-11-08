#!/usr/bin/env python3

"""
Please run in geometry_env:
geometry_env\Scripts\activate
"""

# env vars MUST come before all other imports
import os
from pathlib import Path

# Get absolute path of this script
script_dir = Path(__file__).resolve().parent

print(f"Script Directory: {script_dir}")

# Preparing audios folder
AUDIO_PATH= script_dir / "audios"
os.makedirs(AUDIO_PATH, exist_ok=True)

"""
MLS 数据集 on-the-fly 下载 + 提取一体化脚本（统一随机种子 seed=42）

最终产物（每种语言）：
  - mls_data/<lang>/<lang>.wav          # 该语言合计约 60 分钟的单一 WAV（16 kHz 单声道）
  - mls_data/<lang>/metadata.tsv        # 与 WAV 对应的分段元数据（偏移/说话人/转录）

设计：
  - 不预先下载每种语言的全部语音数据，而是在线流式获取约 60 分钟语音，
    先在临时目录保存独立片段，再用 ffmpeg 合并为单一 WAV，最后删除所有中间文件。
  - 所有语言共用同一个提取函数 extract_language(lang)，仅传入语言名即可。
  - 覆盖：english, german, french, spanish, italian, dutch, portuguese, polish
  - 全部文件保存在本脚本所在目录（mls/）下。

用法：
  python download_extract_mls_on_the_fly.py                 # 全部语言
  python download_extract_mls_on_the_fly.py german english  # 指定语言
  python download_extract_mls_on_the_fly.py --keep-intermediate  # 保留中间文件

依赖：datasets、huggingface_hub、pyarrow、ffmpeg（系统命令）
"""
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

# ═══════ 全局配置（所有语言统一）═══════
SEED = 42                       # ← 统一随机种子
TARGET_SECONDS = 60 * 60        # 每种语言累计采集满 60 分钟即停止
SAVE_ROOT = script_dir / "audios_mls_data"
TARGET_SR = 16000               # 最终 WAV 采样率（与原提取脚本一致）
WAV_CHANNELS = 1

MLS_LANGS = ["german", "french", "spanish", "italian", "dutch", "portuguese", "polish"]
ENGLISH_DATASET = "parler-tts/mls_eng"
MULTILINGUAL_DATASET = "facebook/multilingual_librispeech"

# ═══════ 环境准备 ═══════
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(SAVE_ROOT / "hf_cache")
random.seed(SEED)

from datasets import load_dataset, Audio


def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(s))


def _clean_text(s: str) -> str:
    """TSV 安全化：去掉制表符/换行"""
    return str(s or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _audio_ext(audio_bytes: bytes) -> str:
    if audio_bytes[:4] == b"fLaC":
        return "flac"
    if audio_bytes[:4] == b"OggS":
        return "opus"
    if audio_bytes[:4] == b"RIFF":
        return "wav"
    return "bin"


def _iter_samples(lang: str):
    """统一的数据样本迭代器：非英语用官方 1_hours split；英语用 dev split 本地 parquet。"""
    if lang == "english":
        parquet_path = SAVE_ROOT / "_downloads" / "mls_eng_dev.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if not parquet_path.exists() or parquet_path.stat().st_size < 200 * 1024**2:
            from huggingface_hub import hf_hub_download
            real = hf_hub_download(
                repo_id=ENGLISH_DATASET, repo_type="dataset",
                filename="data/dev-00000-of-00001.parquet",
                local_dir=str(parquet_path.parent))
            src = Path(real)
            if src.resolve() != parquet_path.resolve():
                shutil.copyfile(src, parquet_path)
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        yield from table.to_pylist()
    else:
        ds = load_dataset(MULTILINGUAL_DATASET, lang, split="1_hours", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        yield from ds


def extract_language(lang: str) -> float:
    """统一的提取函数：采集约60分钟语音，合并为单一 WAV + 写 metadata.tsv。返回总秒数。"""
    print(f"\n{'='*55}\n[{lang}] 正在流式获取合计 {TARGET_SECONDS//60} 分钟语音 ...")

    lang_dir = SAVE_ROOT / lang
    tmp_dir = lang_dir / ".tmp_clips"      # 临时片段目录（合并后删除）
    lang_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    # ── 1. 提取片段到临时目录，记录顺序与时长 ──
    clips = []   # 每项: (clip_path, duration, speaker_id, chapter_id, transcript, source_name)
    total_sec = 0.0
    for sample in _iter_samples(lang):
        if total_sec >= TARGET_SECONDS:
            break
        audio_bytes = sample["audio"]["bytes"]
        ext = _audio_ext(audio_bytes)
        speaker_id = _safe_name(str(sample.get("speaker_id", "unknown")))
        chapter_id = _safe_name(str(sample.get("chapter_id", sample.get("book_id", "0"))))
        if sample.get("id"):
            sid = _safe_name(str(sample["id"]))
        else:  # english：没有 id 字段，用 original_path 文件名 + begin_time 保证唯一
            stem = Path(str(sample.get("original_path", "x"))).stem
            sid = _safe_name(f"{stem}_{float(sample.get('begin_time', 0.0)):.2f}")
        src_name = f"{sid}.{ext}"
        clip_path = tmp_dir / f"{len(clips):06d}_{src_name}"
        clip_path.write_bytes(audio_bytes)
        dur = float(sample.get("audio_duration") or
                    (sample.get("end_time", 0) - sample.get("begin_time", 0)) or 0.0)
        clips.append((clip_path, dur, speaker_id, chapter_id,
                      _clean_text(sample.get("transcript", "")), src_name))
        total_sec += dur

    # ── 2. 用 ffmpeg 合并为单一 WAV（16 kHz 单声道）──
    wav_path = lang_dir / f"{lang}.wav"
    concat_list = lang_dir / ".concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for clip_path, *_ in clips:
            f.write(f"file '{clip_path.resolve()}'\n")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-ar", str(TARGET_SR), "-ac", str(WAV_CHANNELS),
        "-c:a", "pcm_s16le", str(wav_path),
    ]
    subprocess.run(cmd, check=True)
    concat_list.unlink(missing_ok=True)

    # ── 3. 写 metadata.tsv（偏移量对应 WAV 内位置）──
    meta_path = lang_dir / "metadata.tsv"
    cum = 0.0
    with open(meta_path, "w", encoding="utf-8") as meta_f:
        meta_f.write("offset_start\toffset_end\tduration\tspeaker_id\tchapter_id\ttranscript\n")
        for clip_path, dur, spk, chap, transcript, src_name in clips:
            start, end = cum, cum + dur
            meta_f.write(f"{start:.3f}\t{end:.3f}\t{dur:.3f}\t{spk}\t{chap}\t{transcript}\n")
            cum = end

    # ── 4. 删除临时片段（保留单一 wav + metadata）──
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[完成] {lang}: {len(clips)} 个片段, {cum/60:.1f} 分钟 → {wav_path.name}")
    return cum


def cleanup_intermediate():
    """删除中间文件（英语 parquet 下载、HuggingFace 缓存等），仅保留最终 WAV 与 metadata。"""
    for name in ("_downloads", "hf_cache"):
        path = SAVE_ROOT / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"[清理] 已删除中间文件：{path}")


def main():

    import sys
    from datetime import datetime
    
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

    # if shutil.which("ffmpeg") is None:
    #     sys.exit("错误：未找到 ffmpeg，请先安装（例如 apt install ffmpeg）。")

    args = sys.argv[1:]
    keep_intermediate = "--keep-intermediate" in args
    langs = [a for a in args if not a.startswith("--")] or (MLS_LANGS + ["english"])

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    (SAVE_ROOT / "hf_cache").mkdir(parents=True, exist_ok=True)

    print(f"随机种子 : {SEED}")
    print(f"保存根目录: {SAVE_ROOT}")
    print(f"目标语言 : {langs}\n")

    results = {}
    for lang in langs:
        try:
            results[lang] = extract_language(lang)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[失败] {lang}: {e}")

    print(f"\n{'='*55}\n全部完成！各语言产物：")
    for name in sorted(os.listdir(SAVE_ROOT)):
        if name in ("hf_cache", "_downloads"):
            continue
        path = SAVE_ROOT / name
        if path.is_dir():
            wav = path / f"{name}.wav"
            size = wav.stat().st_size if wav.exists() else 0
            mins = results.get(name, 0) / 60
            print(f"  {name}/  wav={size/1024**2:.1f} MB, {mins:.1f} 分钟")
            
            # 1: 将 wav 文件从 "english.wav" 重命名为 "english_1_hours_merged.wav"
            wav_new = path / f"{name}_1_hours_merged.wav"
            if wav.exists() and not wav_new.exists():
                wav.rename(wav_new)

            # 2: 将最里层文件夹名修改为 "文件夹名_mls"
            new_path = path.with_name(f"{path.name}_mls")
            if path != new_path:
                path.rename(new_path)
                path = new_path

    # 默认清理中间文件
    if keep_intermediate:
        print("\n[提示] 已通过 --keep-intermediate 保留中间文件")
    else:
        print()
        cleanup_intermediate()

    print(f"Process ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Close log file and restore stdout
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()

if __name__ == "__main__":
    main()
