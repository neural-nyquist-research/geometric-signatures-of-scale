#!/bin/sh
# Restore the unpacked scripts when the data mount hides the copy baked into the image.
#
# The working directory doubles as the mount point people use for their data:
#
#     docker run -v "$BASE_PATH":/home/geometry_user/geometry ...
#
# A bind mount replaces the directory, it does not merge with it, so everything the Dockerfile
# unpacked at /home/geometry_user/geometry disappears behind the host folder. A pristine copy is
# therefore kept at /opt/geometry-scripts, and this entry point copies it back when the mounted
# folder carries no scripts of its own. Files the host already has are left alone, so a host copy
# of the archive still wins.
set -e

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

exec "$@"
