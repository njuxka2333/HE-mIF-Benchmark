# Frozen data splits and catalogs

The files below define the sample assignments and tile catalogs used by the benchmark. Whole-slide images, patch pixels, and extracted feature bags are stored locally and are not included here. Build dataset-specific local assets with `python -m preprocessing.build --dataset <name>` after configuring the data paths.

| Task | Dataset | Files |
|---|---|---|
| Cell semantic segmentation | PathoCell | `cell_semantic_segmentation/pathocell/slide_split.csv` |
| Cell semantic segmentation | Lizard | `cell_semantic_segmentation/lizard/slide_split.csv` |
| Single-cell classification | Immunocto | `single_cell_classification/immunocto/patient_split.csv`, `cell_index.csv.gz`, `other_excluded.csv.gz` |
| Single-cell classification | NuCLS | `single_cell_classification/nucls/fold_1.csv` |
| Prognosis | TCGA-COAD | `prognosis/tcga_coad/slide_split.csv`, `fold_0.csv`–`fold_4.csv` |
| Prognosis | TCGA-LUAD | `prognosis/tcga_luad/slide_split.csv`, `fold_0.csv`–`fold_4.csv` |
| TLS region segmentation | HookNet-TLS | `tls_region_segmentation/hooknet_tls/slide_split.csv`, `patches.csv` |
| Virtual staining | Cell-niches | `virtual_staining/cell_niches/patches.csv` |

The Immunocto patient assignment, per-cell index, and `other`-cell exclusion list are all used by the loader or feature filtering. The NuCLS file contains its locked fold-1 cell assignment. Prognosis fold files preserve the study's five-fold evaluation protocol. The HookNet-TLS and Cell-niches `patches.csv` files preserve the selected tile/FOV geometry. Do not regenerate these files when reproducing the reported evaluation.

For HookNet-TLS tile geometry, counts, and local patch preparation, see [`tls_region_segmentation/hooknet_tls/README.md`](tls_region_segmentation/hooknet_tls/README.md).
