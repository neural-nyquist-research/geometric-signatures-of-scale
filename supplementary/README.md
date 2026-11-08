# Supplementary material

Everything needed to install and run the pipeline on a machine other than the authors'.

| file | what it is |
|---|---|
| `environment_geometry_winpython3.12.9.0.txt` | the **full** pinned environment, torch included, used for extraction and analysis |
| `Dockerfile` | container recipe; builds from the **repository root**, not from this folder |
| `DOCKER_GUIDE.md` | build and run instructions, with the host-mount layout the pipeline expects |
| `scripts.zip` | the script set of **this** version — 65 files, flat — for a machine where the scripts sit beside the data folders |
| `scripts_full.zip` | the same plus the two Praat measurement folders — 138 files, 180 MB — a release asset, not in git, uploaded as eight parts of 22 MB [here](https://github.com/neural-nyquist-research/geometric-signatures-of-scale/releases/tag/v1.0.0) |

## What is in `scripts.zip`

It is deliberately not the whole history. It holds what produces the manuscript as submitted:

| group | files | role |
|---|---|---|
| 1 preparation | 27 | download checkpoints, FLEURS and the Rucheng recording; extract per-layer activations; re-organise the tree; build the typology table |
| 2 MLS layout | 2 | build and unwind the MLS folder structure |
| 3 analysis | 17 | the scripts that compute what the shipped caches hold |
| 4 figures | 7 | the scripts that draw the seven printed figures |
| 5 tables | 12 + `pair_summary.csv` | every table of the manuscript, plus the three cross-checks |
| 6 Praat data | 72 | `praat_PitchTier/` (25 languages) and `praat_PitchTier_mls/` (8): the `.PitchTier` and `.FormantGrid` tracks that `02_02` reads, with the two `.praat` scripts |

`MANIFEST.txt` inside the archive lists every file by name and states what was left out.

**Left out on purpose.** The plotting scripts of the previous version (`01_*`, `03_02`, `04_04`,
`06_*`, `07_*`, `08_*`, `14_*`, `31_*`–`35_*`, `11_02`, `12_02`): every figure of this version was
redrawn, and group 4 is what drew the printed ones. The previous version's table scripts (`43_00` for
its Tab02, `43_02` for its Tab04, `43_03` for its Tab05): those tables are not in the manuscript. Both
sets remain in the repository's `scripts/` folder for provenance. `44_mantel_permutation_test.py` is
absent because it needs the hidden states; its output, `data/mantel_perm_results.csv`, ships instead.

## The Praat folders and the size limit

`scripts_full.zip` carries `praat_PitchTier/` and `praat_PitchTier_mls/` — 910 MB uncompressed, 180 MB
compressed, split into eight parts of 22 MB because GitHub's browser upload stops at 25 MB
(`cat scripts_full.zip.part-* > scripts_full.zip`, md5 `6478086284335081e42ca9824149d2e8`) — because `02_02_formant_dim_analysis_02_praat_sync.py` reads them at `./praat_PitchTier/` and
`./praat_PitchTier_mls/`. GitHub refuses any file above 100 MB, so this archive cannot live in the
repository: it is attached to a release, and the two folders are listed in `.gitignore`. `scripts.zip`
stays in the repository for the common case, since every table and every figure rebuilds without the
Praat tracks.

`Praat_6462.exe` (61 MB in each folder) is not in either archive. `praat_PitchTier/*.txt` records the two
command lines it runs, so any Praat 6.4.x substitutes for it.

## Two layouts, no edits

Unzip beside the data, or use the repository as it stands. Every script resolves its inputs through the
current directory, then its own, then its parent:

```bash
mkdir -p ~/nyquist && cd ~/nyquist
unzip /path/to/scripts.zip          # scripts land beside re-organized/, data/, downstream/, logs/
python 43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check
python fig7_disentanglement.py
```

The same commands run from the repository root with `scripts/` and `figures/` replacing the flat copies.

## Quick start (analysis only)

```bash
python -m venv geometry_env && source geometry_env/bin/activate
pip install -r ../requirements.txt        # analysis-only pins
python ../scripts/43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check
```

For the extraction stage install the full environment instead:

```bash
pip install -r environment_geometry_winpython3.12.9.0.txt
```

## Verified

The archive was extracted into an empty directory with the measurement folders beside it, and every
script that does not need hidden states was run there: 19 of 19 exit 0 (`43_10` exits 4 by design when a
trimmed cache set yields a table that differs from the shipped one), the six tables match the manuscript
byte for byte, and all six figure scripts in the archive reproduce the printed PNGs with matching `md5`.
