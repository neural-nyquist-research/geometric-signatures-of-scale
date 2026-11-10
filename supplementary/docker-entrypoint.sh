#!/bin/sh
# Restore what the data mount hides.
#
# The working directory doubles as the mount point people use for their data:
#
#     docker run -v "$BASE_PATH":/home/geometry_user/geometry ...
#
# A bind mount replaces the directory, it does not merge with it, so everything the Dockerfile
# put at /home/geometry_user/geometry disappears behind the host folder. Pristine copies are
# therefore kept outside it - scripts at /opt/geometry-scripts, the shipped caches, figures and
# tables at /opt/geometry-data - and this entry point copies them back when the mounted folder
# does not already carry them. The Python environment never lives there at all: it is at
# /opt/geometry-env, on PATH, so `python` works without activating anything.
#
# Files the host already has are left alone, so a host copy still wins.
set -e

DATA_DIRS="figures data downstream logs cldf-datasets_wals re-organized re-organized_mls"
MARKER="$PWD/MANIFEST.txt"

if [ ! -e "$MARKER" ] && [ -d /opt/geometry-scripts ]; then
    # -R rather than -a: the stash is root-owned, and preserving ownership as an ordinary user
    # fails on every file even though the copy itself lands.
    cp -R /opt/geometry-scripts/. "$PWD"/ 2>/dev/null || true
    if [ -e "$MARKER" ]; then
        echo "[entrypoint] $PWD had no scripts; restored them from /opt/geometry-scripts" \
             "($(find "$PWD" -maxdepth 1 -name '*.py' | wc -l) python files)"
    else
        echo "[entrypoint] WARNING: cannot write into $PWD, so the scripts stay hidden." \
             "Re-run with a writable mount, or unzip supplementary/scripts.zip there yourself." >&2
    fi
fi

if [ -d /opt/geometry-data ]; then
    for d in $DATA_DIRS; do
        if [ ! -e "$PWD/$d" ]; then
            cp -R "/opt/geometry-data/$d" "$PWD/$d" 2>/dev/null || true
        fi
    done
    missing=""
    for d in $DATA_DIRS; do
        [ -e "$PWD/$d" ] || missing="$missing $d"
    done
    if [ -z "$missing" ]; then
        echo "[entrypoint] shipped caches, figures and tables in place: $DATA_DIRS"
    else
        echo "[entrypoint] WARNING: could not restore:$missing (read-only mount?)" >&2
    fi
fi

exec "$@"
