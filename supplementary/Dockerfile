# Container recipe for the geometric-signatures pipeline.
#
# Build from the REPOSITORY ROOT, because `requirements.txt` and the script
# archive `supplementary/scripts.zip` are taken from there:
#
#     docker build -f supplementary/Dockerfile -t geometry-analysis:latest .
#
# The image carries code and the Python environment only. Audio, checkpoints and
# activations arrive through a bind mount at /home/geometry_user/geometry; see
# DOCKER_GUIDE.md. The archive is unpacked into the working directory, so the
# scripts sit beside re-organized/ rather than in a scripts/ folder, and are restored at
# start-up if a bind mount over the working directory hides them.

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

RUN useradd -m -s /bin/bash geometry_user \
    && mkdir -p /home/geometry_user/geometry \
    && chown -R geometry_user:geometry_user /home/geometry_user/geometry

WORKDIR /home/geometry_user/geometry

COPY --chown=geometry_user:geometry_user requirements.txt ./
COPY --chown=geometry_user:geometry_user supplementary/environment_geometry_winpython3.12.9.0.txt ./

RUN python -m venv geometry_env \
    && ./geometry_env/bin/pip install --upgrade pip \
    && ./geometry_env/bin/pip install --no-cache-dir \
         -r environment_geometry_winpython3.12.9.0.txt

# Copy the scripts and requirements
COPY --chown=geometry_user:geometry_user supplementary/scripts.zip ./

# Unzip scripts and cleanup. A pristine copy is kept at /opt/geometry-scripts: the data mount
# replaces the working directory at run time, so the unpacked copies here are not always visible.
RUN mkdir -p /opt/geometry-scripts \
    && unzip -o -q scripts.zip -d /opt/geometry-scripts \
    && unzip -o scripts.zip \
    && rm scripts.zip \
    && chown -R root:root /opt/geometry-scripts \
    && chmod -R a+rX /opt/geometry-scripts
COPY --chown=geometry_user:geometry_user figures ./figures
COPY --chown=geometry_user:geometry_user data ./data
COPY --chown=geometry_user:geometry_user downstream ./downstream
COPY --chown=geometry_user:geometry_user logs ./logs
COPY --chown=geometry_user:geometry_user cldf-datasets_wals ./cldf-datasets_wals
COPY --chown=geometry_user:geometry_user README.md LICENSE ./

# Measurement caches are ~5.6 MB and are copied in, so the container can
# reproduce every table without a host mount. Extraction outputs go to the
# mounted volume at run time.
COPY --chown=geometry_user:geometry_user re-organized ./re-organized
COPY --chown=geometry_user:geometry_user re-organized_mls ./re-organized_mls

RUN mkdir -p audios huggingface models inferred results docs \
    && chown -R geometry_user:geometry_user /home/geometry_user/geometry

# Restores the scripts into the working directory when a data mount has hidden them.
COPY --chown=root:root supplementary/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 755 /usr/local/bin/docker-entrypoint.sh

USER geometry_user

# Entry point
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
