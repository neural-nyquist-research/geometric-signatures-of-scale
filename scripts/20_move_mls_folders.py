#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move every folder whose name contains "_mls" (along with all of its contents)
from the source tree into a sibling destination tree, preserving the relative
parent structure.

Example
-------
source layout:  re-organized/mHuBERT-147/hidden_states_00/dutch_mls/<...>
result layout: re-organized_mls/mHuBERT-147/hidden_states_00/dutch_mls/<...>

Usage:
    python move_mls_folders.py                 # real run against ./re-organized
    python move_mls_folders.py --dry-run       # preview only, no changes
    python move_mls_folders.py --src ./x --dst ./x_mls
"""

import argparse
import os
import shutil
import sys


def find_mls_dirs(src, dst, marker, case_sensitive=False):
    """Collect absolute paths of dirs under src whose name contains marker.

    When an `_mls` dir is found it is returned and its subtree is not
    descended into (moving the parent moves everything below it). The
    destination is never descended into, even if it nests under src.
    """
    found = []
    dst_abs = os.path.abspath(dst)
    normalize = (lambda n: n) if case_sensitive else (lambda n: n.lower())

    for root, dirs, files in os.walk(src):
        root_abs = os.path.abspath(root)
        keep = []
        for d in dirs:
            full = os.path.join(root_abs, d)
            if full == dst_abs or full.startswith(dst_abs + os.sep):
                continue  # skip the destination itself
            if marker in normalize(d):
                found.append(full)
            else:
                keep.append(d)
        dirs[:] = keep
    return found


def main():
    ap = argparse.ArgumentParser(description="Move *_mls folders preserving tree.")
    ap.add_argument("--src", default="re-organized",
                    help="source folder to walk (default: ./re-organized)")
    ap.add_argument("--dst", default="re-organized_mls",
                    help="destination root (default: ./re-organized_mls)")
    ap.add_argument("--marker", default="_mls",
                    help="substring that marks a folder to move (default: _mls)")
    ap.add_argument("--case-sensitive", action="store_true",
                    help="match the marker case-sensitively (default: off)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be done without moving anything")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    if not os.path.isdir(src):
        sys.exit(f"ERROR: source folder not found: {src}")

    if src == dst or dst.startswith(src + os.sep):
        sys.exit("ERROR: destination must not be inside or equal to the source.")

    marker = args.marker
    cs = args.case_sensitive
    matches = find_mls_dirs(src, dst, marker, cs)

    print(f"Source : {src}")
    print(f"Dest   : {dst}")
    print(f"Marker : '{marker}'  (case-{'in' if not cs else ''}sensitive)")
    print("-" * 70)
    print(f"Found {len(matches)} folder(s) whose name contains '{marker}'.")
    if not matches:
        sys.exit("Nothing to do.")

    moved = 0
    for full in matches:
        rel = os.path.relpath(full, src)          # e.g. mHuBERT-147/hidden_states_00/dutch_mls
        target = os.path.join(dst, rel)
        if args.dry_run:
            print(f"  WOULD MOVE: re-organized\\{rel}")
            continue
        # Create the parent chain *except* the final _mls dir, so shutil.move
        # renames the whole source dir onto target (same basename). If target
        # already exists this raises FileExistsError instead of merging.
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(full, target)                 # move whole dir -> target
        print(f"  moved: {rel}")
        moved += 1

    if args.dry_run:
        print("\nDry run complete -- no changes made.")
    else:
        print(f"\nFinished: moved {moved} folder(s).")


if __name__ == "__main__":
    main()