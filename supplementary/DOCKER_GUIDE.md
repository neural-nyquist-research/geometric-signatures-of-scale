# Docker Deployment Guide

The container pins Python 3.12.9, FFmpeg and the scientific stack, so the pipeline behaves the same on
any host. Build from the **repository root** — the Dockerfile lives in `supplementary/`, but its build
context is the root, because it copies `requirements.txt`, the environment file, `supplementary/scripts.zip`
and the Rucheng recording.

## 1. Build

```bash
cd /path/to/geometric-signatures-of-scale
git lfs pull                                        # the recording is 108 MB and tracked by Git LFS
docker build -f supplementary/Dockerfile -t geometry-analysis:latest .
```

The image installs the pinned environment into a virtual environment at
`/home/geometry_user/geometry/geometry_env`, unpacks the script archive into `/home/geometry_user/geometry`
itself — no `scripts/` folder in between — and copies the shipped caches, figures, tables, logs and the
Rucheng recording beside them.

The build context is roughly 120 MB, almost all of it the recording. `.dockerignore` keeps the git history,
`results/` and the two 61 MB Windows Praat binaries out of it.

## 2. Prepare host storage

The extraction stage produces ~1 TB of activations. Create the folders you are going to mount on the host
first, or the container will write them as root and you will fight permissions later.

```bash
export BASE_PATH="/mnt/bigdrive/geometry_project"
mkdir -p $BASE_PATH/{audios,huggingface,models,inferred,results}
```

## 3. Run

```bash
docker run --rm -it \
  --memory=64g --shm-size=8g \
  -v "$BASE_PATH/audios":/home/geometry_user/geometry/audios \
  -v "$BASE_PATH/huggingface":/home/geometry_user/geometry/huggingface \
  -v "$BASE_PATH/models":/home/geometry_user/geometry/models \
  -v "$BASE_PATH/inferred":/home/geometry_user/geometry/inferred \
  -v "$BASE_PATH/results":/home/geometry_user/geometry/results \
  geometry-analysis:latest
```

**Mount folders, not the working directory.** A bind mount replaces the directory it is mounted on instead
of merging with it, so `-v "$BASE_PATH":/home/geometry_user/geometry` would hide the scripts, the
environment and every shipped cache behind whatever the host folder holds. The five mounts above are the
folders the pipeline writes into; everything else stays as the image built it.

**Adapt the limits to the host.** `--memory` should sit near 80 % of physical RAM (32 GB host → `--memory=26g`);
`--shm-size` of 4–8 GB is enough for the multiprocessing in the extraction scripts. If the container is
killed during extraction, reduce `BATCH_LAYERS` / `BATCH_LAYERS_OVERRIDE` in `00_15`–`00_23` first — the
defaults assume 128 GB.

## 4. Inside the container

```bash
source geometry_env/bin/activate
python -V                                                                # 3.12.9, the pinned environment
python 43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check   # analysis, minutes
./00_15-18_infer.sh natural                                              # extraction, hours
```

No `scripts/` prefix: the archive is flat, and the scripts are addressed from the working directory. The
same holds for the batch runners `00_15-18_infer.sh` and `00_15-18_infer.bat`. `00_02` copies
`hn_rucheng_1_hours_merged.wav` from the working directory into `audios/_hn_rucheng/`, where the
`00_15`–`00_23` extraction scripts read audio.

### Regenerating the measurement caches

`re-organized/` and `re-organized_mls/` ship inside the image and are **not** mounted above, so the shipped
caches stay visible and every table reproduces without a host mount. If you re-run the analysis stage
(`02_01`–`02_05`) and want the result to outlive the container, copy the shipped caches out once and mount
them back:

```bash
docker run --rm -v "$BASE_PATH":/out geometry-analysis:latest \
  sh -c 'cp -a /home/geometry_user/geometry/re-organized /home/geometry_user/geometry/re-organized_mls /out/'

docker run --rm -it \
  --memory=64g --shm-size=8g \
  -v "$BASE_PATH/audios":/home/geometry_user/geometry/audios \
  -v "$BASE_PATH/huggingface":/home/geometry_user/geometry/huggingface \
  -v "$BASE_PATH/models":/home/geometry_user/geometry/models \
  -v "$BASE_PATH/inferred":/home/geometry_user/geometry/inferred \
  -v "$BASE_PATH/results":/home/geometry_user/geometry/results \
  -v "$BASE_PATH/re-organized":/home/geometry_user/geometry/re-organized \
  -v "$BASE_PATH/re-organized_mls":/home/geometry_user/geometry/re-organized_mls \
  geometry-analysis:latest
```

Mounting a cache folder that is still empty would hide the shipped copy, which is why the first command
seeds the host folder.

## 5. Notes

- The image carries the code, the environment and the measurement caches: no FLEURS or MLS audio, no
  checkpoints, no activations. Those arrive through the mounts or are downloaded by the `00_*` scripts.
- GPU images need a different base (CUDA or ROCm); the shipped Dockerfile is CPU-only. Extraction on CPU
  is supported and is what the authors' timings refer to.
- Analysis needs no GPU and runs comfortably inside 8 GB of RAM.
- To move the image to another machine: `docker save -o geometry_image_export.tar geometry-analysis:latest`
  and `docker load -i geometry_image_export.tar`.
