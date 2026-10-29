# HookNet-TLS split and tile catalog

`slide_split.csv` assigns slides to train, valid, and test. `patches.csv` is the frozen tile catalog. Each row describes a 256 µm tile (1024 px at 0.25 µm/px); the `x_he`, `y_he`, `w_he`, `h_he`, and `he_mpp` columns identify its source crop in the whole-slide image.

The catalog contains 21,537 train, 3,716 valid, and 7,053 test tiles from 537, 77, and 153 slides, respectively. Rest tiles use `rest_keep_fraction=0.01` and `random_seed=42`; TLS/GC tiles are retained in full.

Whole-slide images and tile pixels are not included in this repository. After configuring the HookNet-TLS image and annotation paths, build the tile images with:

```bash
python -m preprocessing.build --dataset hooknet_tls
```

The generated images and masks are written under `$HE_MIF_DATASETS_ROOT/HookNet_TLS_fixed_grid_ps1024/`; the frozen `patches.csv` catalog is copied into that directory for training.
