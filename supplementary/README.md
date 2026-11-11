# Supplementary material

Everything needed to install and run the pipeline on a machine other than the authors'.

| file | what it is |
|---|---|
| `environment_geometry_winpython3.12.9.0.txt` | the **full** pinned environment, torch included, used for extraction and analysis |
| `Dockerfile` | container recipe; builds from the **repository root**, not from this folder |
| `DOCKER_GUIDE.md` | build and run instructions, with the host-mount layout the pipeline expects |
| `scripts.zip` | the contents of the repository's `scripts/` folder — 93 files, flat, 274 KB — for a machine where the scripts sit beside the data folders |
| `.dockerignore` | keeps the build context at ~120 MB: no git history, no `results/`, no Windows Praat binaries |

## What is in `scripts.zip`

It is `scripts/` as it stands, with the archive root holding the scripts themselves — no `scripts/`
folder in between, so it unpacks into the working directory. 93 files:

| group | files | role |
|---|---|---|
| preparation | 27 | download checkpoints, FLEURS and the Rucheng recording; extract per-layer activations; re-organise the tree; build the typology table |
| MLS layout | 2 | build and unwind the MLS folder structure |
| measurement chain | 5 | `02_01`–`02_05`: formant tracks, the Praat sync, the correlation loops that produce the shipped caches |
| analysis and case study | 8 | `02_06` (Fig. 2), `02_07` (the depth test), `04_01`–`04_03`, `05_`, `11_01`, `12_01` (the Rucheng case study) |
| tables and checks | 12 | every table of the manuscript, plus the four cross-checks |
| Mantel re-run | 1 | `44_mantel_permutation_test.py`; needs hidden states, so its output ships in `data/` instead |
| legacy plots, for provenance | 29 | the plots of the previous version: `01_*`, `03_*`, `04_04`, `06_*`–`14_*`, `31_*`–`35_*` |
| Praat runbooks | 6 | `praat_PitchTier/` and `praat_PitchTier_mls/`: `pitchTier.praat`, `formantGrid.praat`, `run.txt` |
| case-study inputs | 3 | two small arrays `12_01` reads, and `pair_summary.csv`, the input of the pair-contrast table |

`MANIFEST.txt` inside the archive lists every file by name and states what was left out.

**Left out on purpose.** The two `Praat_6462.exe` Windows binaries (61 MB each; `run.txt` records the
command line and any Praat 6.4.x works). `hn_rucheng_1_hours_merged.wav` (108 MB): a recording rather than
code, stored with **Git LFS** in `scripts/` — run `git lfs install` and `git lfs pull` to fetch it. And the
previous version's table scripts `43_00` (its Tab02), `43_02` (its Tab04) and `43_03` (its Tab05), because
none of those three tables is in the manuscript; they stay in `scripts/`.

The legacy plots travel with the archive but read per-language arrays that the caches do not carry: they are
there for provenance, not for re-running. The figures of this version are drawn by `figures/` (group 4 in
the main README), and the numbers in the manuscript come from the table scripts.

## The Praat runbooks

`praat_PitchTier/` and `praat_PitchTier_mls/` carry `pitchTier.praat`, `formantGrid.praat` and `run.txt`,
not the tracks those scripts produce. `run.txt` says what to do: copy the language wav files into the
folder and run its two lines. The `.PitchTier` and `.FormantGrid` files (910 MB) appear beside them.

They are kept that way because nothing in the archive needs the tracks: `02_02` reads them once to build
the formant caches, and those caches ship in `re-organized/`. `Praat_6462.exe` is left out as well (63 MB,
Windows only); any Praat 6.4.x runs the two lines unchanged.

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

The archive was extracted into an empty directory with the measurement folders beside it, and the scripts
that the manuscript's numbers come from were run there: the twelve table and check scripts, `02_06`, the
`02_07` depth test, and the seven figure scripts. All exit 0 except `43_10`, which exits 4 by design when a
trimmed cache set yields a table that differs from the shipped one; the tables match the manuscript byte for
byte, and the figure scripts reproduce the printed PNGs with matching `md5`.
