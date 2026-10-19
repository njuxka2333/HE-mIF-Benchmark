# HookNet-TLS frozen catalog

`hooknet-tls_split.csv` is the slide-level train/valid/test assignment.

`patches.csv` is the paper tile catalog. Geometry columns match Cell-niches:

`x_he, y_he, w_he, h_he, he_mpp, size_um, stride_um, he_relpath, patch_id`

Each row is one 256 µm cell (1024 px at 0.25 µm/px). `x_he..h_he` / `he_mpp` are
the WSI level-0 source crop; the stored PNG is resampled to 1024 px.
Rest tiles were kept at `rest_keep_fraction=0.01` with `random_seed=42`; TLS/GC
tiles were kept in full.

| split | tiles | slides |
|-------|------:|-------:|
| train | 21537 | 537 |
| valid | 3716 | 77 |
| test | 7053 | 153 |

PNG/mask pixels are not in git. From `benchmark/`, crop them from the WSIs
using this catalog:

```bash
python -m preprocessing.build --dataset hooknet_tls
```

That writes `$HE_MIF_DATASETS_ROOT/HookNet_TLS_fixed_grid_ps1024/` and copies
this `patches.csv` next to the PNGs. Training reads that dataset-root catalog.
`scripts/rebuild_hooknet_tls_canonical_grid.py` re-plans from XML; do not use
it for a paper rerun.
