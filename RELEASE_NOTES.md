# v1.0.0 — reproduction package

Code, measurement caches and figures for the manuscript *Geometric Signatures of Scale in the XLS-R
Family: Typological Alignment, Manifold Change, and Expert Separation in Self-Supervised Speech Models*.

## Assets

| file | size | what it is |
|---|---|---|
| **`scripts_full.zip.part-0` … `part-3`** | 45 + 45 + 45 + 36 MB | the complete pipeline (65 scripts) **plus** the two Praat measurement folders, `praat_PitchTier/` (25 languages) and `praat_PitchTier_mls/` (8). Uploaded in four parts because GitHub's browser upload stops at 25 MB per file; join them before unzipping |
| *source archive* | 11 MB | the repository itself — clone it instead |

```bash
cat scripts_full.zip.part-0 scripts_full.zip.part-1 scripts_full.zip.part-2 scripts_full.zip.part-3 > scripts_full.zip
md5sum scripts_full.zip      # 6478086284335081e42ca9824149d2e8
unzip scripts_full.zip       # 138 files
```

The archive is here rather than in the repository because of GitHub's 100 MB per-file limit. The
Praat folders are 910 MB uncompressed; `02_02_formant_dim_analysis_02_praat_sync.py` reads them at
`./praat_PitchTier/`, so unzip beside the other data folders.

`Praat_6462.exe` is not in the archive. `praat_PitchTier/*.txt` records the two command lines it runs, so
any Praat 6.4.x substitutes.

## Reproduce the paper in minutes

```bash
pip install -r requirements.txt
python scripts/43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check
python scripts/43_07_table_downstream_cer_tab-downstream_cer.py --check
python figures/fig7_disentanglement.py
```

The seven tables come out byte-identical to the manuscript's, and so do the seven figures. The
measurement caches that make this possible (7.5 MB) are in the repository.

## What the package covers

- seven self-supervised encoders: XLSR-53, XLS-R-300M/1B/2B, MMS-300M/1B, mHuBERT-147;
- 24 FLEURS languages plus the Rucheng dialect, with MLS (8 languages) as a second corpus;
- a late alignment checkpoint at relative depth 0.75–0.79, a saturating step in intrinsic dimension
  between the 300M and 1B/2B tiers, disjoint tone and syllable weights at Layer 38, and no ordering of
  recognition accuracy by any geometric quantity.

## Licence

GPL-3.0. Typological features derive from WALS and PHOIBLE (CC BY-SA 4.0 and CC BY-SA 3.0 respectively);
the audio is not redistributed.
