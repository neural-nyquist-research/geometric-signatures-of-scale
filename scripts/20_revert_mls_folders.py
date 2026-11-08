#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revert the operation performed by move_mls_folders.py.

move_mls_folders.py moved every folder whose name contains "_mls" from
    re-organized/<model>/hidden_states_XX/<lang>_mls/<...>
into a sibling tree preserving the relative structure:
    re-organized_mls/<model>/hidden_states_XX/<lang>_mls/<...>

This script undoes that: it walks re-organized_mls, finds every *_mls folder
(with everything below it), and moves it back into re-organized, restoring the
exact original path.

Example
-------
revert layout: re-organized_mls/mHuBERT-147/hidden_states_00/dutch_mls/<...>
  ->restored  re-organized/mHuBERT-147/hidden_states_00/dutch_mls/<...>

Usage:
    python revert_mls_folders.py                 # real run
    python revert_mls_folders.py --dry-run       # preview only, no changes
    python revert_mls_folders.py --src ./x_mls --dst ./x
"""

import argparse
import os
import shutil
import sys


def find_mls_dirs(src, marker="_mls", case_sensitive=False):
    """Collect absolute paths of *_mls dirs (including nested ones) under src.

    When a *_mls dir is found it is returned and its subtree is not descended
    into (moving the parent moves everything below it), mirroring the forward
    script's behaviour. The nested case is handled naturally: reverting the
    parent is enough.
    """
    found = []
    normalize = (lambda n: n) if case_sensitive else (lambda n: n.lower())
    seen = set()  # guard against duplicate inserts if src ever contains dst

    for root, dirs, files in os.walk(src):
        root_abs = os.path.abspath(root)
        keep = []
        for d in dirs:
            full = os.path.join(root_abs, d)
            if full in seen:
                continue
            if marker in normalize(d):
                found.append(full)
                seen.add(full)
            else:
                keep.append(d)
        dirs[:] = keep
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Reverse move_mls_folders.py: put *_mls folders back.")
    ap.add_argument("--src", default="re-organized_mls",
                    help="the *_mls tree to walk back from (default: ./re-organized_mls)")
    ap.add_argument("--dst", default="re-organized",
                    help="destination to restore into (default: ./re-organized)")
    ap.add_argument("--marker", default="_mls",
                    help="substring that identifies a moved folder (default: _mls)")
    ap.add_argument("--case-sensitive", action="store_true",
                    help="match the marker case-sensitively (default: off)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be done without moving anything")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    if not os.path.isdir(src):
        sys.exit(f"ERROR: source folder not found: {src}")

    if src == dst or dst.startswith(src + os.sep) or src.startswith(dst + os.sep):
        sys.exit("ERROR: source and destination must be independent folders.")

    marker = args.marker
    cs = args.case_sensitive
    matches = find_mls_dirs(src, marker, cs)

    print(f"Restore into : {dst}")
    print(f"Walking from : {src}")
    print(f"Marker       : '{marker}'  (case-{'in' if not cs else ''}sensitive)")
    print("-" * 70)
    print(f"Found {len(matches)} folder(s) whose name contains '{marker}' to move back.")
    if not matches:
        sys.exit("Nothing to do.")

    restored = 0
    for full in matches:
        rel = os.path.relpath(full, src)          # e.g. <model>/<hs>/dutch_mls
        target = os.path.join(dst, rel)
        if args.dry_run:
            print(f"  WOULD RESTORE: {rel}")
            continue
        # Rebuild the parent chain on the destination side. Fail loudly if the
        # exact target already exists (it should not, since it was moved out).
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target) and not _empty_dir(target):
            print(f"  SKIP (target already exists): {target}")
            continue
        if os.path.exists(target):
            # target exists but empty -> remove it so move can take place
            shutil.rmtree(target)
        shutil.move(full, target)
        print(f"  restored: {rel}")
        restored += 1

    if args.dry_run:
        print("\nDry run complete -- no changes made.")
    else:
        print(f"\nFinished: restored {restored} folder(s).")


def _empty_dir(path):
    try:
        return os.path.isdir(path) and not os.listdir(path)
    except OSError:
        return False


if __name__ == "__main__":
    main()