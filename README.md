# Geometric Signatures of Scale in the XLS-R Family

**Typological Alignment, Manifold Change, and Expert Separation in Self-Supervised Speech Models**

Junjie He, Fang Wang — Shaanxi Normal University

Code, measurement caches and figures behind the manuscript submitted to *IEEE/ACM Transactions on Audio,
Speech, and Language Processing*. The repository has two uses: **reproduce every table and figure in
the paper in minutes from the shipped caches**, or **regenerate the whole study from public checkpoints
and corpora**, which takes days and a large disk.

---

## 1. What the study does

Seven self-supervised encoders — XLSR-53, XLS-R-300M, XLS-R-1B, XLS-R-2B, MMS-300M, MMS-1B and
mHuBERT-147 — on 24 FLEURS languages plus the Rucheng dialect, with MLS (8 languages) as a second
corpus. The paper reports four results:

| result | where in the paper |
|---|---|
| A Mantel correlation against WALS/PHOIBLE places a **late alignment checkpoint at relative depth 0.75–0.79** in every XLS-R model | §IV-A, Fig. 1, Table I |
| **Intrinsic dimension rises in one step** between the 300M and 1B/2B tiers (12/12 rounds, effect size 2.5) and does not rise further from 1B to 2B; no other geometric quantity holds a direction | §IV-B, Tables III–IV |
| At Layer 38 of XLS-R-2B the **tone and syllable-structure discriminants share no dimension** (top-20 intersection 0, Jaccard 0.00 %) | §IV-D, Figs. 6–7 |
| A frozen-feature CTC probe across all seven encoders finds **no ordering by intrinsic dimension** (Spearman ρ = −0.04) | §IV-F, Table VII |

The claims are deliberately correlational: the paper reports *signatures* of scale, not the origin of a
scaling law.

---

## 2. Reproduce the paper (minutes, no GPU)

The measurement caches are shipped (≈7.5 MB of CSV), so every table rebuilds and every figure regenerates
without touching audio or model weights.

```bash
pip install -r requirements.txt          # Python 3.12

# Tables I, III, IV, V, VI, VII, VIII
python scripts/43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py --check
python scripts/43_01_table_scaling_metrics_Tab03_tab-scaling_metrics.py
python scripts/43_11_table_pair_contrasts.py --check
python scripts/43_00b_table_alignment_comparison_4models.py --check
python scripts/43_04_table_cross_family_tab-cross_family.py --check
python scripts/43_07_table_downstream_cer_tab-downstream_cer.py --check
python scripts/43_08_table_anchor_vs_late_tab-anchor_vs_late.py --check

# The seven figures, from the shipped measurements
python figures/fig1_mantel.py
python scripts/02_06_formant_dim_analysis_06_corr-loop_04_Fig02_fig-manifold_comparison.py
python figures/fig3_functional_hierarchy.py
python figures/fig4_temporal_entropy.py
python figures/fig5_id_language_paths.py
python figures/fig6_lda_projection.py
python figures/fig7_disentanglement.py

# Cross-checks: corpus completeness, the Section IV-G p-values, the ±2 s lag window
python scripts/43_06_corpus_check_fleurs_vs_mls.py
python scripts/43_09_check_section_IV_G_numbers.py
python scripts/43_12_check_lag_window_numbers.py
```

Table scripts write into `results/`; figure scripts overwrite `figures/`. Six of the table scripts take
`--check`, which recomputes the values and exits non-zero if they drift from the numbers printed in the
paper; `43_06`, `43_09` and `43_12` exist only to catch that drift.

Verified in this layout, not assumed. The six tables come out byte-identical to the manuscript's — `md5`
006e7c3e, 26180cb2, 1f6cce22, 6326a21e, c95d1272, 88cd897a for the alignment, anchor-versus-late,
cross-family, downstream, Mantel and pair-contrast tables — and so do all seven figures: `47ab6c8a`,
`1bca8cb7`, `2b193cfb`, `4980b19b`, `d7a1f686`, `4f8c6766`, `e88ee77c`. These are the redrawn figures of
this version, drawn from the shipped measurements, not the earlier draft plots.

Two scripts need inputs that are not in the repository. `scripts/44_mantel_permutation_test.py` re-runs
the 9999-permutation Mantel null of §III-E and needs the hidden states; without them it stops with a
message and points at `data/mantel_perm_results.csv`, the shipped output of the run the paper reports.
`scripts/43_10_pairwise_capacity_pairs.py` needs the full per-node caches behind the twelve-round scan:
run against the trimmed caches it will compute a different table, so it writes to `results/` and exits 4
when its output differs from the shipped `scripts/pair_summary.csv`, which is the table `43_11` reads.

---

## 3. Regenerate from scratch (days, ≥1 TB)

Everything in the shipped caches is produced by the numbered pipeline. Run from the repository root, in
order, with the environment of `supplementary/environment_geometry_winpython3.12.9.0.txt`.

| stage | scripts | what it does |
|---|---|---|
| **00** data preparation | `00_00_dl_models.py`, `00_01_download_fleurs.py`, `00_02_copy_hn_rucheng.py` | fetch the seven checkpoints, the FLEURS audio, and place the Rucheng recording |
| **00** inference | `00_15`–`00_18` (XLS-R family), `00_21`–`00_23` (MMS-300M, MMS-1B, mHuBERT-147) | extract per-layer activations as `X_flat_stable_1hours.npy`, one hour of speech per language |
| **00** re-organisation | `00_31_data_copy_reorganize.py` (and `…_reverse.py`) | turn the language-centric tree into the per-layer tree the analysis reads |
| **00** typology | `00_98_*`, `00_99_*` | build the WALS/PHOIBLE feature table |
| **01–13** FLEURS analysis | `01_*` … `13_*` | Mantel profile, formant/`F0` correlations, entropy, LDA discriminants, ripser persistence, the Rucheng case, and the figures built from them |
| **20** MLS preparation | `20_move_mls_folders.py`, `20_revert_mls_folders.py` | build and unwind the MLS folder layout |
| **31–35** MLS analysis | `31_*` … `35_*` | the same analyses on the eight-language MLS corpus |
| **43–44** manuscript | `43_*`, `44_*` | every table and check in the paper, from the caches above |

Batch runners for the long stages: `00_15-18_infer.{sh,bat}`, `00_21-23_infer.{sh,bat}`,
`00_33_re-organize.{sh,bat}`, with `_mls` and `_reverse` variants beside them.

Figures come out of `figures/*.py`, not out of the plotting scripts inside `01`–`35`: those drew the
earlier drafts and stop at the measurement stage now.

**Hardware.** Extraction is the expensive part: ~1 TB of disk and, in the authors' runs, 128 GB of RAM
with `BATCH_LAYERS` tuned for it — lower it in `00_15`–`00_23` on a smaller machine. Analysis needs only
a laptop CPU.

---

## 4. Layout, and what ships

```
scripts/             the numbered pipeline (see §3) + pair_summary.csv
figures/             the seven paper figures and the scripts that draw them
data/                typological vectors, LDA weights/projection, Mantel permutation results
cldf-datasets_wals/  the same typology table at the path the older scripts hard-code
re-organized/        FLEURS measurement cache, seven encoders
re-organized_mls/    MLS measurement cache, same protocol
downstream/          per-language CER of the frozen-feature CTC probe (30 epochs, 8 layers)
logs/                the three Mantel runs whose p-values §IV-G quotes
results/             table-script outputs (created on first run, gitignored)
supplementary/       Docker recipe, pinned environment, and the two script archives
```

The measurement caches are ~7.5 MB of CSV and are what makes the paper reproducible on a laptop. Each is
the analysis result of the stage that produced it, not raw audio. Model directories keep the checkpoint
names the pipeline writes (`wav2vec2-large-xlsr-53`, `wav2vec2-xls-r-300m`, `wav2vec2-xls-r-1b`,
`wav2vec2-xls-r-2b`, `mms-300m`, `mms-1b`, `mHuBERT-147`):

| shipped file | rows / size | read by |
|---|---|---|
| `re-organized/<model>/formant_dim_analysis_cache/Formant_Lag_Analysis_Results_moreLayers.csv` | per language × layer × dimension | Figs. 2–5, Tables III–IV, VII |
| `re-organized/<model>/ID_result/ID_Analysis_L*_Fixed.csv` | one file per layer, 49 for the 49-layer encoders | Figs. 3 and 5, `04_02`, `43_04` |
| `re-organized/<model>/global_results/Layer_Evolution_Analysis_allLayers.csv` | per language × layer × dimension, 14 files, 1.8 MB | Fig. 3 |
| `re-organized/manifold_area.csv`, `mean_shift.csv`, `mantel_results_cache.csv` | one row per model–layer | Fig. 2, Table I |
| `re-organized/Entropy_Results_*_moreLayers.csv`, `Raw_Lag_Dist_*_moreLayers.csv` | per model × layer | Fig. 4, §IV-C |
| `re-organized_mls/…` | the same set for the eight MLS languages, 2.0 MB | MLS replication (§IV-E) |
| `downstream/asr.csv` | 7 encoders × 24 languages | Table VII |
| `logs/anchor_13_13_25_25.log`, `late_checkpoint.log`, `control_19_19_38_38.log` | three run logs | `43_09` |
| `data/target_languages_typology.csv` | 25 languages × 7 columns | Tables I, V; Figs. 6–7 |
| `data/lda_projection_L38.csv` | 25 rows | Fig. 6 |
| `data/lda_weights_L38.csv` | 1,920 rows | Fig. 7 |
| `data/mantel_perm_results.csv` | 148 model–layers | §III-E |
| `figures/fig1…fig7*.png` | the seven paper figures as printed | — |

## 5. Table and figure map

| paper element | script |
|---|---|
| Table I, Mantel profile | `43_05_table_mantel_checkpoints_tab-mantel-checkpoints.py` |
| Table III, geometry at the anchor | `43_01_table_scaling_metrics_Tab03_tab-scaling_metrics.py` |
| Table IV, capacity contrasts | `43_11_table_pair_contrasts.py` (reads `scripts/pair_summary.csv`) |
| Table V, alignment at two checkpoints | `43_00b_table_alignment_comparison_4models.py` |
| Table VI, seven encoders | `43_04_table_cross_family_tab-cross_family.py` |
| Table VII, downstream CER | `43_07_table_downstream_cer_tab-downstream_cer.py` |
| Table VIII, anchor versus checkpoint | `43_08_table_anchor_vs_late_tab-anchor_vs_late.py` |
| Fig. 1, Mantel profile | `figures/fig1_mantel.py` |
| Fig. 2, manifold at the anchor | `scripts/02_06_formant_dim_analysis_06_corr-loop_04_Fig02_fig-manifold_comparison.py` |
| Fig. 3, segmental versus suprasegmental | `figures/fig3_functional_hierarchy.py` |
| Fig. 4, temporal entropy | `figures/fig4_temporal_entropy.py` |
| Fig. 5, intrinsic dimension across depth | `figures/fig5_id_language_paths.py` |
| Fig. 6, tone-discriminant axis | `figures/fig6_lda_projection.py` (reads `data/lda_projection_L38.csv`) |
| Fig. 7, tone versus syllable weights | `figures/fig7_disentanglement.py` (reads `data/lda_weights_L38.csv`) |

Two revision scripts produce supporting tables that are **not printed** in the final manuscript:
`43_02_table_id_ht_comparison_Tab04_…py` (intrinsic dimension and entropy per model–layer) and
`43_03_table_id_comparison_Tab05_…py`. `43_10_pairwise_capacity_pairs.py` regenerates
`scripts/pair_summary.csv`, the input of Table IV; it needs the per-node caches, so its output ships and
a run against the trimmed caches exits 4 rather than quietly replacing the shipped table.

`scripts/` also keeps the plotting scripts of the previous version — `01_*`, `03_02`, `04_04`, `06_*`,
`07_*`, `08_*`, `14_*` and the MLS pair `31_*`–`35_*`. They drew the earlier drafts and are superseded by
the `figures/` scripts above; they stay for provenance and are not part of `scripts.zip`.

## 6. What is not here

In the repository: the Praat tracks (`praat_PitchTier/`, `praat_PitchTier_mls/`, 910 MB). They ship with
the whole script set in one archive, 180 MB, split into four parts on the release page — GitHub's browser
upload stops at 25 MB, so it travels in pieces:

<https://github.com/neural-nyquist-research/geometric-signatures-of-scale/releases/tag/v1.0.0>

```bash
cat scripts_full.zip.part-0 scripts_full.zip.part-1 scripts_full.zip.part-2 scripts_full.zip.part-3 \
    > scripts_full.zip
md5sum scripts_full.zip      # 6478086284335081e42ca9824149d2e8
unzip scripts_full.zip       # 138 files
```

Unzip it beside `re-organized/`, and `02_02` finds the tracks where it expects them.

Not anywhere: the audio (FLEURS, MLS, the Rucheng recording), the extracted hidden states (gigabytes per
encoder) and the model checkpoints. All are regenerated by the `00_*` scripts from public sources;
`hn_rucheng_1_hours_merged.wav` must be obtained separately. Only `02_02` needs the Praat tracks, and the
shipped formant caches already hold its output, so no table or figure depends on them.

## 7. Environment

Python 3.12.9. `requirements.txt` pins the analysis environment; the full extraction environment, including
torch, is in `supplementary/environment_geometry_winpython3.12.9.0.txt`. Three installation routes —
Windows/WinPython, Linux/macOS with pyenv, and Docker — are in `supplementary/README.md` and
`supplementary/DOCKER_GUIDE.md`.

If matplotlib cannot import numpy from the interpreter you think you are using, check `PYTHONPATH`
before reinstalling: an inherited path from another interpreter produces the misleading
`No module named 'numpy._core._multiarray_umath'`, which is a path problem, not a broken install.

## 8. Scope of the claims

The repository reproduces a saturating step in intrinsic dimension, a late alignment checkpoint, and
disjoint tone and syllable weight sets. It does **not** support a power law, a monotone trend across the
four geometric quantities, or a link between geometry and recognition accuracy.

## 9. Citation

```bibtex
@article{he2026geometric,
  title   = {Geometric Signatures of Scale in the {XLS-R} Family: Typological Alignment,
             Manifold Change, and Expert Separation in Self-Supervised Speech Models},
  author  = {He, Junjie and Wang, Fang},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2026},
  note    = {Under review}
}
```

## 10. Licence

GNU General Public License v3.0 — see `LICENSE`.
