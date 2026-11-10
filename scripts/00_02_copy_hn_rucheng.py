import os
import shutil
from pathlib import Path

# Get absolute path of this script
script_dir = Path(__file__).resolve().parent

print(f"Script Directory: {script_dir}")

os.makedirs(str(Path(f"{script_dir}/audios")), exist_ok=True)
os.makedirs(str(Path(f"{script_dir}/audios/_hn_rucheng")), exist_ok=True)

hn_rucheng_filename="hn_rucheng_1_hours_merged.wav"

# The recording sits beside the script in the repository layout, and beside it again once
# scripts.zip is unpacked flat; the scripts/ subfolder is kept as a further candidate so that
# an older tree, or a copy of this file left at the project root, still resolves.
candidates = [
    Path(f"{script_dir}/scripts/{hn_rucheng_filename}"),
    Path(f"{script_dir}/{hn_rucheng_filename}"),
    Path(f"{script_dir.parent}/scripts/{hn_rucheng_filename}"),
]
src_path=str(next((p for p in candidates if p.exists()), candidates[0]))
dst_path=str(Path(f"{script_dir}/audios/_hn_rucheng/{hn_rucheng_filename}"))

if not os.path.exists(dst_path):
    try:
        shutil.copy2(src_path, dst_path)
        print(f"✅ hn_rucheng copied: {src_path}")
    except Exception as e:
        print(f"\n❌ Error: Cannot copy {src_path}")
else:
    print(f"✅ hn_rucheng already existed: {src_path}")