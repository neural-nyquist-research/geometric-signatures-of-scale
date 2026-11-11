# Container recipe for the geometric-signatures pipeline.
#
# Build from the REPOSITORY ROOT, because `requirements.txt`, the environment file and the
# script archive are taken from there:
#
#     docker build -f supplementary/Dockerfile -t geometry-analysis:latest .
#
# The image carries the code, the pinned environment and the shipped measurement caches. Audio,
# checkpoints and activations arrive through bind mounts, one mount per data folder - never a
# mount over the working directory itself, because that would hide everything the image put
# there. See DOCKER_GUIDE.md.
#
#   mounted at run time : audios/, huggingface/, models/, inferred/, results/
#   from the image      : the scripts, geometry_env/, figures/, data/, downstream/, logs/,
#                         cldf-datasets_wals/, re-organized/, re-organized_mls/ and
#                         hn_rucheng_1_hours_merged.wav

FROM python:3.12.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        unzip \
        build-essential \
        libsndfile1 \
        sox \
        git \
        nano \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create the user and the working directory, and hand both to it, before anything is copied in.
RUN useradd -m -s /bin/bash geometry_user \
    && mkdir -p /home/geometry_user/geometry \
    && chown -R geometry_user:geometry_user /home/geometry_user/geometry

WORKDIR /home/geometry_user/geometry

USER geometry_user

COPY --chown=geometry_user:geometry_user requirements.txt ./
COPY --chown=geometry_user:geometry_user supplementary/environment_geometry_winpython3.12.9.0.txt ./

# The environment lives in the working directory and is entered the usual way:
#     source geometry_env/bin/activate
# Mounting the data folders one by one keeps it visible; mounting the whole working directory
# would not.
RUN python -m venv geometry_env \
    && ./geometry_env/bin/pip install --upgrade pip \
    && ./geometry_env/bin/pip install --no-cache-dir \
         -r environment_geometry_winpython3.12.9.0.txt

# Copy the scripts and requirements
COPY --chown=geometry_user:geometry_user supplementary/scripts.zip ./

# Unzip scripts and cleanup
RUN unzip -o scripts.zip && rm scripts.zip

COPY --chown=geometry_user:geometry_user figures ./figures
COPY --chown=geometry_user:geometry_user data ./data
COPY --chown=geometry_user:geometry_user downstream ./downstream
COPY --chown=geometry_user:geometry_user logs ./logs
COPY --chown=geometry_user:geometry_user cldf-datasets_wals ./cldf-datasets_wals
COPY --chown=geometry_user:geometry_user README.md LICENSE ./

# Measurement caches are ~7.5 MB and are copied in, so the container can reproduce every table
# without a host mount. Extraction outputs go to the mounted folders at run time.
COPY --chown=geometry_user:geometry_user re-organized ./re-organized
COPY --chown=geometry_user:geometry_user re-organized_mls ./re-organized_mls

# The Rucheng recording is 108 MB, so it is not inside scripts.zip; 00_02 copies it from the
# working directory into audios/_hn_rucheng/. The repository tracks it through Git LFS, so an
# unpulled checkout would bake in a 130-byte pointer - the check below says so.
COPY --chown=geometry_user:geometry_user scripts/hn_rucheng_1_hours_merged.wav ./
RUN size=$(stat -c%s hn_rucheng_1_hours_merged.wav) && \
    if [ "$size" -lt 1000000 ]; then \
        echo "WARNING: hn_rucheng_1_hours_merged.wav is $size bytes, which is a Git LFS pointer." && \
        echo "         Run 'git lfs pull' in the repository and build again." ; \
    fi

RUN mkdir -p audios huggingface models inferred results docs

# Entry point
CMD ["/bin/bash"]
