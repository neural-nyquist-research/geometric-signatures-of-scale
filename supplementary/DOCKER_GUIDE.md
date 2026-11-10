# Docker Deployment Guide

The container pins Python 3.12.9, FFmpeg and the scientific stack, so the pipeline behaves the same on
any host. Build from the **repository root** — the Dockerfile lives in `supplementary/`, but its build
context is the root, because it copies `requirements.txt` and `supplementary/scripts.zip`.

## 1. Build

```bash
cd /path/to/geometric-signatures-of-scale
docker build -f supplementary/Dockerfile -t geometry-analysis:latest .
```

The image installs the pinned environment into a virtualenv at **`/opt/geometry-env`**, which is on `PATH`,
so `python` is 3.12.9 with the pins in every shell and nothing has to be activated. The script archive is
unpacked into `/home/geometry_user/geometry` itself, without a `scripts/` folder in between: the scripts sit
next to `re-organized/` and the other data folders, and every command runs from the working directory.

## 2. Prepare host storage

The extraction stage produces ~1 TB of activations. Create the mount points on the host first, or the
container will write them as root and you will fight permissions later.

```bash
export BASE_PATH="/mnt/bigdrive/geometry_project"
mkdir -p $BASE_PATH/{audios,huggingface,models,inferred,re-organized,re-organized_mls,downstream,results}
cp scripts/hn_rucheng_1_hours_merged.wav "$BASE_PATH"
```

The scripts and the shipped caches need no copying: the mount hides whatever the image put at that path, so
the container restores them itself on start-up (§4), about 12 MB once. The one file that has to come from
outside is the Rucheng recording — it is 108 MB and travels with the repository through Git LFS rather than
inside `scripts.zip`, so `00_02` (which copies it to `audios/_hn_rucheng/`) needs it in the mount. Without it
everything except the case study still runs.

## 3. Run

```bash
docker run --rm -it \
  --memory=64g --shm-size=8g \
  -v "$BASE_PATH":/home/geometry_user/geometry \
  geometry-analysis:latest
```

**Adapt the limits to the host.** `--memory` should sit near 80 % of physical RAM (32 GB host → `--memory=26g`);
`--shm-size` of 4–8 GB is enough for the multiprocessing in the extraction scripts. If the container is
killed during extraction, reduce `BATCH_LAYERS` / `BATCH_LAYERS_OVERRIDE` in `00_15`–`00_23` first — the
defaults assume 128 GB.

## 4. Inside the container

```bash
python -V                                                                # 3.12.9, already the pinned env
python 43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check   # analysis, minutes
./00_15-18_infer.sh natural                                              # extraction, hours
```

There is no `source geometry_env/bin/activate` step: the environment is at `/opt/geometry-env` and already
first on `PATH`. `source /opt/geometry-env/bin/activate` works if you want the prompt changed, and
`pip install` inside it is writable by `geometry_user`.

No `scripts/` prefix: the archive is flat, and the scripts are addressed from the working directory. The
same holds for the batch runners `00_15-18_infer.sh` and `00_15-18_infer.bat`.

### Why the scripts, caches and environment are still there

The working directory doubles as the mount point, and a bind mount **replaces** the directory instead of
merging with it: everything the image put at `/home/geometry_user/geometry` disappears behind `$BASE_PATH`.
That is why nothing the container needs lives there alone:

| what | where it really lives | how it reaches the working directory |
|---|---|---|
| Python environment | `/opt/geometry-env` | on `PATH`; never hidden, nothing to activate |
| scripts | `/opt/geometry-scripts` | entry point copies them in when the folder has no `MANIFEST.txt` |
| `figures/`, `data/`, `logs/`, `downstream/`, `cldf-datasets_wals/`, `re-organized/`, `re-organized_mls/` | `/opt/geometry-data` | entry point copies each missing folder in, once (~12 MB) |

Files the host already has are left untouched, so your own copy of a cache or a script wins; run the
container on a read-only mount and the entry point warns instead of failing. If you would rather keep the
scripts on the host between runs, unzip `supplementary/scripts.zip` into `$BASE_PATH` yourself.

## 5. Notes

- The image contains **code only**: no audio, no checkpoints, no activations. Those arrive through the
  mount at `$BASE_PATH` or are downloaded by the `00_*` scripts.
- GPU images need a different base (CUDA or ROCm); the shipped Dockerfile is CPU-only. Extraction on CPU
  is supported and is what the authors' timings refer to.
- Analysis needs no GPU and runs comfortably inside 8 GB of RAM.
