# Frozen splits

These CSVs are the reported train/valid/test (and CV fold) assignments. Do not
regenerate the **case** tables for a paper rerun. Patch pixels and CLAM coord
bags are not in git: `python -m preprocessing.build --dataset …` writes them
under `$HE_MIF_DATASETS_ROOT`.

| task | files |
|------|--------|
| Single-cell | `single_cell_cls/immunocto/immunocto_patient_split.csv`, `immunocto_train_other_excluded.csv.gz`; `single_cell_cls/nucls/nucls_cell_split_fold1.csv` |
| Prognosis | `prognosis_os/tcga_coad_split.csv`, `tcga_coad_split_shazam_fold{0-4}.csv`, `tcga_luad_split.csv`, `tcga_luad_split_cv5_os_balanced_fold{0-4}.csv` |
| TLS | `tls_seg/hooknet-tls_split.csv` (slide split) plus `tls_seg/patches.csv` (tile crop boxes used by `preprocessing.build`) |
| Cell semantic | `cell_semantic_seg/phenocell_dataset_split.csv`, `lizard_split.csv` |
| Virtual staining | `virtual_staining/patches.csv` (Cell-niches 256 µm patches, with HE/mIF boxes) |

Immunocto `cell_index.csv` and `immunocto_train_other_excluded.csv` are over
GitHub's 100 MB limit. Git tracks the `.csv.gz` twins; the loader reads either.
NuCLS fold-1 is the locked split.

Case-table builders (not the paper rerun): `build_tcga_coad_split.py`,
`build_luad_cv5_os_balanced.py`, `canonicalize_immunocto_split.py`.
