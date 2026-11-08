# Docker Deployment Guide

The container pins Python 3.12.9, FFmpeg and the scientific stack, so the pipeline behaves the same on
any host. Build from the **repository root** — the Dockerfile lives in `supplementary/`, but its build
context is the root, because it copies `scripts/` and `requirements.txt`.

## 1. Build

```bash
cd /path/to/geometric-signatures-of-scale
docker build -f supplementary/Dockerfile -t geometry-analysis:latest .
```

The image installs `requirements.txt` into a virtual environment at
`/home/geometry_user/geometry/geometry_env` and copies the pipeline to `/home/geometry_user/geometry/scripts`.

## 2. Prepare host storage

The extraction stage produces ~1 TB of activations. Create the mount points on the host first, or the
container will write them as root and you will fight permissions later.

```bash
export BASE_PATH="/mnt/bigdrive/geometry_project"
mkdir -p $BASE_PATH/{audios,huggingface,models,inferred,re-organized,re-organized_mls,downstream,results}
```

## 3. Run

```bash
docker run --rm -it \
  --memory=64g --shm-size=8g \
  -v "$BASE_PATH":/home/geometry_user/geometry \
  -v /path/to/geometric-signatures-of-scale/scripts:/home/geometry_user/geometry/scripts:ro \
  geometry-analysis:latest
```

**Adapt the limits to the host.** `--memory` should sit near 80 % of physical RAM (32 GB host → `--memory=26g`);
`--shm-size` of 4–8 GB is enough for the multiprocessing in the extraction scripts. If the container is
killed during extraction, reduce `BATCH_LAYERS` / `BATCH_LAYERS_OVERRIDE` in `00_15`–`00_23` first — the
defaults assume 128 GB.

## 4. Inside the container

```bash
source geometry_env/bin/activate
python scripts/43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check   # analysis, minutes
./00_15-18_infer.sh natural                                                     # extraction, hours
```

The batch runners assume the flat layout; if you prefer them, unzip `supplementary/scripts.zip` into the
working directory and run from there.

## 5. Notes

- The image contains **code only**: no audio, no checkpoints, no activations. Those arrive through the
  mount at `$BASE_PATH` or are downloaded by the `00_*` scripts.
- GPU images need a different base (CUDA or ROCm); the shipped Dockerfile is CPU-only. Extraction on CPU
  is supported and is what the authors' timings refer to.
- Analysis needs no GPU and runs comfortably inside 8 GB of RAM.
