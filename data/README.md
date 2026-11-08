# data/

Small derived tables. Everything here is produced by a script in the manuscript repository; nothing
is hand-edited.

`target_languages_typology.csv` also sits in `../cldf-datasets_wals/`, byte for byte, because the
figure scripts resolve it through that path. Both copies are shipped so neither needs a patch.

## `target_languages_typology.csv` — 25 rows × 7 columns

The per-language typological vectors used by the Mantel analysis, and the table that appeared as the
appendix in earlier versions of the manuscript.

| column | meaning |
|---|---|
| `Language_Name` | language as used in the paper |
| `ISO639P3code` | ISO 639-3 code |
| `Syllable Structure` | WALS feature, normalised to [0, 1] |
| `Tone` | tone value, normalised to [0, 1]; the analysis binarises at `> 0.5` |
| `Consonant Inventories` | inventory size, normalised |
| `Vowel Quality Inventories` | inventory size, normalised |
| `voiced_ratio` | proportion of voiced segments, normalised |

Sources: WALS (Dryer & Haspelmath, eds.) and PHOIBLE (Moran & McCloy, eds.), via the CLDF datasets.

Two coding decisions worth knowing:

- **Tone threshold is `> 0.5`, not `>= 0.5`.** Japanese sits exactly at 0.500 and is therefore
  non-tonal; a `>=` rule would move it into the tonal class and change the discriminant fit.
- **Rucheng carries the ISO code `hak`.** Rucheng (汝城) is a Sinitic variety usually placed with the
  Xiang or Hakka group, and ISO 639-3 has no dedicated code for it. `hak` is the closest available
  label, not a claim about classification, and the feature values in that row follow the description
  of the dialect in the source corpus rather than the Hakka entry of WALS.

## `lda_projection_L38.csv` — 25 rows

| column | meaning |
|---|---|
| `language` | language name |
| `iso` | ISO 639-3 code |
| `tone` | binary tone label used by the discriminant (1 = tonal) |
| `lda1` | projection on the first tone-discriminant axis at Layer 38 of XLS-R-2B |

Vietnamese is the far pole (+4.05), Rucheng second (+3.87); the classes separate with a 0.76 margin
and no error. Figure 6 is drawn from this file.

## `lda_weights_L38.csv` — 1,920 rows

One row per hidden dimension of Layer 38 (XLS-R-2B).

| column | meaning |
|---|---|
| `dimension` | index, 0–1919 |
| `tone_weight` | L2 norm of the dimension's coefficient in the tone discriminant |
| `syllable_weight` | L2 norm of the dimension's coefficient in the syllable-structure discriminant |

The 20 largest `tone_weight` and the 20 largest `syllable_weight` dimensions share no member
(Jaccard 0.00%), which is the claim Figure 7 illustrates. Recomputing it:

```python
import csv, numpy as np
rows = list(csv.DictReader(open("lda_weights_L38.csv")))
t = np.array([float(r["tone_weight"]) for r in rows])
s = np.array([float(r["syllable_weight"]) for r in rows])
a, b = set(np.argsort(t)[-20:]), set(np.argsort(s)[-20:])
print(len(a & b), len(a & b) / len(a | b))     # -> 0 0.0
```

## `mantel_perm_results.csv` — 148 rows

The permutation test behind Section III-E, one row per model–layer (49 + 49 + 25 + 25).

| column | meaning |
|---|---|
| `Model` | model key, matching `re-organized/mantel_results_cache.csv` |
| `Layer` | transformer layer index |
| `N_lang` | languages in the Mantel matrix (24 FLEURS languages; the dialect is not in this analysis) |
| `r_M` | standardised Mantel statistic, identical to the Pearson correlation over the 276 distance pairs |
| `p_one_sided` | $9999$ permutations of $D^{\text{typ}}$ rows and columns, positive direction |
| `p_two_sided` | two-sided permutation $p$ |
| `r_partial`, `p_partial` | partial Mantel controlling the per-language count of retained dimensions at that layer |

`r_M` reproduces the canonical column of `mantel_results_cache.csv` to $10^{-6}$ on all 148 layers, which
is what fixes the typological matrix as the Euclidean one over the five unit-scaled features.

The partial columns are filled for 68 of the 148 layers. The paper does **not** report them: a
retained-dimension count is a property of the layer rather than of a language pair, so a Mantel partial
correlation on it is only defined when the count varies across languages, and in the formant cache it
varies too little on most layers to mean anything.

