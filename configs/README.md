# Configuration files (YAML only)

This directory holds **YAML manifests** for datasets, models, and experiments.

| File / folder | Purpose |
|---------------|---------|
| `datasets.yaml` | Production dataset paths and split columns |
| `datasets_test.yaml` | Pytest dataset registry (same keys as production) |
| `benchmark_models.yaml` | ROSIE / HEMIT / GigaTIME / MIPHEI-ViT settings |
| `extraction/protocol.yaml` | Canonical layers, geometry, and Pre_Features split layout |
| `experiments/*.yaml` | Per-run extraction **and** training feature pointers |
| `downstream/` | One YAML per reported run (`{task}/{dataset}/{encoder,output}/{model}.yaml`) |

Python loaders live elsewhere:

| Module | Role |
|--------|------|
| `data/loader.py` | Dataset registry and split CSV loading |
| `utils/experiment_config.py` | Experiment YAML parsing |
| `models/schema.py`, `models/registry.py` | Model schema and YAML registry |
| `utils/checkpoints.py` | Local / Hugging Face checkpoint loading |
| `utils/paths.py` | `REPO_ROOT`, `CONFIG_DIR`, `checkpoints/` |
| `utils/hf_auth.py` | Hugging Face token from `.env` |
| `utils/output_layout.py` | Output path templates |

Import loaders from `data.loader`, `utils.experiment_config`, `models.registry`, or `utils` — not from this folder.

Feature extraction is one entry point:

```bash
python -m feature_extraction.extract --config configs/experiments/abmil_tcga_coad_rosie_40x_s128.yaml
python -m feature_extraction.extract --dataset immunocto --model uni2 --layers encoder
```

`configs/extraction/protocol.yaml` controls which encoder/output layers to write and whether the dataset uses `train/valid/test` or `fold_0..fold_4` (valid=test). Experiment YAMLs still name the run, model, and training `feat_dir`. `--layers all` extracts encoder and mIF in one forward. TLS / PathoCell / Lizard / Cell-niches are online and have no Pre_Features bags.

Train a reported winner from the repo root (`PYTHONPATH` includes `benchmark/`):

```bash
python -m training.pipeline --task prognosis --models rosie
python -m training.train --config configs/downstream/cellseg/pathocell/encoder/hemit.yaml
python -m training.train --config configs/downstream/virtual_staining/cell_niches/output/rosie.yaml
```

A train command writes under `Evaluation/` (or that YAML's `run.output_dir`).
`results/` is the frozen paper snapshot. Relocate images with `HE_MIF_DATASETS_ROOT`.
Cell-niches raw FOVs: `HE_MIF_CELL_NICHES_RAW`. Paper reproduction steps:
repository `README.md`. Split-table builders live in `benchmark/scripts/`
and are not the paper rerun.
