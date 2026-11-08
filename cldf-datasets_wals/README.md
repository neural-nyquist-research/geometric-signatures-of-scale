# cldf-datasets_wals

The pipeline resolves the typological vectors through this path, which is how the folder arrived from the
CLDF download (`00_98`, `00_99`). The file is byte-identical to `../data/target_languages_typology.csv`,
which is what the `43_*` table scripts read; both are shipped so the figure scripts and the table scripts
work without editing paths.

`target_languages_typology.csv` holds the WALS and PHOIBLE features for the 25 languages of the study, one
row per language: tone inventory size, syllable-structure class, and the five continuous features used in
the Mantel distance.
