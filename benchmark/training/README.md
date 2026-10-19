# `training/`

Task entrance is `python -m training.pipeline`. One YAML is
`python -m training.train`. Do not start a paper rerun from `Evaluation/`.

```bash
python -m training.pipeline --task virtual_staining   # one task
python -m training.train --config configs/downstream/.../uni2.yaml
```

This folder is **load / train / score entry only**. Models, metrics, and
datasets live in sibling packages:

| Path | Role |
|------|------|
| `training/` | Router, loops, YAML config, augmentations |
| `heads/` | MIL / UNETR / cellseg / mIF decoders |
| `metrics/` | Classification, survival, segmentation, vstain, clinical stats |
| `losses/` | Train losses |
| `data/` | WSI/FOV loaders **and** extracted feature bags |
| `mif/` | Generated-mIF trainers + Cell-niches scorer (live generator, not in `results/`) |
| `models/mif_output.py` | Shared frozen HE→mIF forward used by scoring and mIF training |

| File in `training/` | Role |
|------|------|
| `train.py` | Router: YAML task → offline / online / mIF train / mIF score |
| `pipeline.py` | All arms of one task, then `postprocess.py`, then example tiles |
| `example_tiles.py` | VS CD3 panel / TLS crop / PathoCell crop after a task finishes |
| `postprocess.py` | Cross-run tables (DeLong, FOV bootstrap, fold CI) |
| `offline_train.py` | Archived bag loop (SCC, OS) |
| `online_segmentation_train.py` | Archived TLS / cellseg loop |

Bytecode that still says `from training.heads import …` is remapped in
`training/__init__.py`. The loops are archived; do not recut them if you want
the published numbers.

## Archived bytecode

A few PrePATH modules never had recoverable `.py`. Those still load from `_orig_pyc/`:

- `offline_train.py`, `data/feature_bags.py`, `heads/` (factory), `metrics/` (core), `online_segmentation_train.py`
- helpers that online bytecode imports from `training.train` (`train.pyc`)

Do not recut those loops if you want the published numbers.
