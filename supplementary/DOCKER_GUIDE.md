# Docker Deployment Guide

The container pins Python 3.12.9, FFmpeg and the scientific stack, so the pipeline behaves the same on
any host. Build from the **repository root** — the Dockerfile lives in `supplementary/`, but its build
context is the root, because it copies `requirements.txt` and `supplementary/scripts.zip`.

## 1. Build

```bash
cd /path/to/geometric-signatures-of-scale
docker build -f supplementary/Dockerfile -t geometry-analysis:latest .
```

The image installs `requirements.txt` into a virtual environment at
`/home/geometry_user/geometry/geometry_env` and unpacks the script archive into `/home/geometry_user/geometry`
itself, without a `scripts/` folder in between: inside the container the scripts sit next to `re-organized/`
and the other data folders, and every command runs from the working directory.

## 2. Prepare host storage

The extraction stage produces ~1 TB of activations. Create the mount points on the host first, or the
container will write them as root and you will fight permissions later.

```bash
export BASE_PATH="/mnt/bigdrive/geometry_project"
mkdir -p $BASE_PATH/{audios,huggingface,models,inferred,re-organized,re-organized_mls,downstream,results}
unzip -o supplementary/scripts.zip -d "$BASE_PATH"    # the mount hides the copy baked into the image
```

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
source geometry_env/bin/activate
python 43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check   # analysis, minutes
./00_15-18_infer.sh natural                                              # extraction, hours
```

No `scripts/` prefix: the archive is flat, and the scripts are addressed from the working directory. The
same holds for the batch runners `00_15-18_infer.sh` and `00_15-18_infer.bat`.

## 5. Notes

- The image contains **code only**: no audio, no checkpoints, no activations. Those arrive through the
  mount at `$BASE_PATH` or are downloaded by the `00_*` scripts.
- GPU images need a different base (CUDA or ROCm); the shipped Dockerfile is CPU-only. Extraction on CPU
  is supported and is what the authors' timings refer to.
- Analysis needs no GPU and runs comfortably inside 8 GB of RAM.
