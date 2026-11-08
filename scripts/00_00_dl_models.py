import os
from huggingface_hub import snapshot_download
from pathlib import Path

# Get absolute path of this script
script_dir = Path(__file__).resolve().parent

print(f"Script Directory: {script_dir}")

# Huggingface Token
#os.environ["HF_TOKEN"] = "your_hugging_face_token_here"

# Mirror & Local cache
#os.environ["HF_ENDPOINT"]           = "https://hf-mirror.com"
os.environ["HF_HOME"]               = str(Path(f"{script_dir}/huggingface"))
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(f"{script_dir}/huggingface/hub"))

# Preparing models folder
MODEL_PATH= script_dir / "models"
os.makedirs(MODEL_PATH, exist_ok=True)

print(f"Local Model Directory: {MODEL_PATH}")
print(str(Path(f"{script_dir}/huggingface")))
print(str(Path(f"{script_dir}/huggingface/hub")))

# Performing downloads
print(f"\n" + "="*70)
print("Fetching model (1/8): facebook/wav2vec2-large-xlsr-53")
print(f"\n" + "="*70)
snapshot_download(
    repo_id="facebook/wav2vec2-large-xlsr-53",
    local_dir=str(Path(f"{MODEL_PATH}/wav2vec2-large-xlsr-53")),
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n" + "="*70)
print("Fetching model (2/8): facebook/wav2vec2-xls-r-300m")
print(f"\n" + "="*70)
snapshot_download(
    repo_id="facebook/wav2vec2-xls-r-300m",
    local_dir=str(Path(f"{MODEL_PATH}/wav2vec2-xls-r-300m")),
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n" + "="*70)
print("Fetching model (4/8): facebook/wav2vec2-xls-r-2b")
print(f"\n" + "="*70)
snapshot_download(
    repo_id="facebook/wav2vec2-xls-r-2b",
    local_dir=str(Path(f"{MODEL_PATH}/wav2vec2-xls-r-2b")),
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n" + "="*70)
print("Download complete.")
print(f"\n" + "="*70)