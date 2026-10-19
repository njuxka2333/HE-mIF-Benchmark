# `scripts/`

These are **split-table and dataset-prep helpers**, not the paper rerun.
A clone should use:

```bash
python -m preprocessing.build --dataset …
python -m feature_extraction.extract …
python -m training.pipeline --task …
```

`preprocessing.build` already calls `build_lizard_pathocell_hdf.py` and
`build_immunocto_webdataset.py` for those dataset names.

| Script | When you would run it |
|--------|------------------------|
| `build_tcga_coad_split.py`, `build_tcga_coad_cv_splits.py`, `build_luad_cv5_os_balanced.py` | Recut a frozen case table (do not do this for a paper rerun) |
| `canonicalize_immunocto_split.py`, `place_immunocto.py` | Rebuild the leakage-free Immunocto placement after a new extract |
| `cut_cell_niches_paired_patches.py`, `rescore_vstain_from_4um_cache.py` | One-off VS catalog / cache work |
| `build_hooknet_tls.py`, `rebuild_hooknet_tls.py` | Older TLS grid builders; official TLS tiles come from `preprocessing.build --dataset hooknet_tls` |
