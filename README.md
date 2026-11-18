# HE–mIF Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2611.19905-b31b1b.svg)](https://arxiv.org/abs/2611.19905)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **[Exact Paper Title]** ([arXiv:2611.19905](https://arxiv.org/abs/2611.19905)).

This repository reproduces our benchmark of hematoxylin and eosin (H&E) to multiplex immunofluorescence (mIF) virtual staining and foundation models across **six pathology models** and **five downstream tasks**. Frozen data splits are versioned in `splits/`, experiment recipes are stored in `configs/`, and executable pipelines reside in `benchmark/`.

## Repository structure

```text
.
├── benchmark/              # Preprocessing, feature extraction, training, and evaluation pipelines
├── configs/                # Dataset registries and task/model experiment recipes
├── splits/                 # Frozen train/val/test split tables and tile manifests
├── checkpoints/            # Local directory for pretrained model weights (git-ignored)
└── requirements.txt        # Pinned Python package dependencies
```

## Tasks and datasets

All experiments use frozen patient- or slide-level splits under `splits/` to help prevent data leakage. Raw whole-slide images (WSIs) and patches must be obtained from their original public sources due to redistribution terms:

| Task | Dataset(s) | Evaluation metric(s) |
|---|---|---|
| Virtual staining | Cell-niches | CD3 Spearman correlation, AUPRC |
| Single-cell classification | Immunocto, NuCLS | Linear-probe AUROC |
| Survival / prognosis | TCGA-COAD, TCGA-LUAD | 5-fold cross-validated C-index |
| TLS region segmentation | HookNet-TLS | Macro F1 |
| Cell semantic segmentation | PathoCell, Lizard | Instance-level macro F1 |

## Requirements and setup

### Hardware and software environment

Experiments were conducted on Linux (`Ubuntu 22.04`, `CUDA [11.8 / 12.1]`) using NVIDIA `[e.g., A100 80GB / RTX 4090 24GB]` GPUs. Use Python 3.10 or 3.11 and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

### Upstream model repositories

Several model adapters depend on their upstream source implementations. Clone the required repositories at the project root (`UNI2-h` loads directly via Hugging Face and does not require a source checkout):

```bash
git clone https://gitlab.com/enable-medicine-public/rosie.git rosie
mkdir -p HEMIT
git clone https://github.com/BianChang/Pix2pix_DualBranch.git HEMIT/Pix2pix_DualBranch
git clone https://github.com/prov-gigatime/GigaTIME.git GigaTIME
git clone https://github.com/Sanofi-Public/MIPHEI-ViT.git MIPHEI-ViT
git clone https://github.com/ratschlab/HistoPlexer.git HistoPlexer
git clone https://github.com/MrPeterJin/ASlide ASlide
```

### Environment variables and pretrained weights

Add `benchmark/` and upstream `ASlide` to `PYTHONPATH`, and point `HE_MIF_DATASETS_ROOT` to your local dataset root:

```bash
export PYTHONPATH="$PWD/benchmark:$PWD/ASlide${PYTHONPATH:+:$PYTHONPATH}"
export HE_MIF_DATASETS_ROOT=/path/to/datasets
# Override individual dataset locations if stored outside HE_MIF_DATASETS_ROOT:
# export PREPATH_DATASET_IMMUNOCTO_DIR=/path/to/ImmunoctoV1
# Source FOVs are required only when rebuilding paired Cell-niches patches:
# export HE_MIF_CELL_NICHES_RAW=/path/to/Cell-niches-data
```

`HE_MIF_DATASETS_ROOT` expects the directory names defined in `configs/datasets.yaml`, unless overridden by `PREPATH_DATASET_<NAME>_DIR`.

Place pretrained model checkpoints in `checkpoints/` (or set `HE_MIF_CKPT_DIR` to a custom path). Note that `UNI2-h`, `GigaTIME`, and the `H-optimus-0` backbone require gated Hugging Face access—authenticate via `huggingface-cli login` or export `HF_TOKEN` prior to running.

| Model | Local checkpoint filename | Official weight source |
|---|---|---|
| UNI2-h | `uni2_pytorch_model.bin` | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| GigaTIME | `gigatime.pth` | [prov-gigatime/GigaTIME](https://huggingface.co/prov-gigatime/GigaTIME) |
| MIPHEI-ViT | `mipheivit.safetensors` | [MIPHEI-ViT v1.0.0 release](https://github.com/Sanofi-Public/MIPHEI-ViT/releases/tag/v1.0.0) |
| HEMIT | `hemit_v1.pth` | Official Google Drive (`1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd`) |
| ROSIE | `rosie_best_model_single.pth` | [ericwu09/ROSIE](https://huggingface.co/ericwu09/ROSIE) |
| HistoPlexer | `histoplexer.pt` | `HistoPlexer/demo/model/pytorch_model.pt` (upstream repo) |

## Data preparation

This repository distributes frozen split tables and tile catalogs, excluding raw WSIs, image patches, and pre-extracted features. After configuring dataset paths, run the preprocessing pipeline for the desired datasets:

```bash
python -m preprocessing.build --dataset hooknet_tls
python -m preprocessing.build --dataset cell_niches
python -m preprocessing.build --dataset tcga_coad
```

Run only the preparation commands required for your target task. Dataset-specific preprocessing protocols and frozen catalogs are documented in the corresponding `splits/` subdirectories.

## Running the benchmark

Execute a single model configuration or an entire task suite from the repository root:

```bash
# Train and evaluate a single configuration
python -m training.train \
  --config configs/downstream/cell_semantic_segmentation/pathocell/encoder/hemit.yaml

# Run task-level pipelines
python -m training.pipeline --task single_cell_classification
python -m training.pipeline --task prognosis --models uni2
python -m training.pipeline --task virtual_staining --significance
```

Use `python -m training.pipeline --all` to reproduce the full benchmark suite. Append `--route encoder` or `--route output` to evaluate a specific representation route (latent encoder features vs. synthesized mIF outputs), and `--test-only` to run inference on existing downstream checkpoints.

For offline WSI tasks (e.g., ABMIL survival prediction), extract patch-level feature bags first using the shared extraction protocol:

```bash
python -m feature_extraction.extract \
  --config configs/experiments/abmil_tcga_coad_rosie_40x_s128.yaml
```

Evaluation metrics and logs are saved to `Evaluation/` by default, and extracted feature bags are written to `Pre_Features/`.

## Configuration and split layout

* **Configs:** `configs/downstream/<task>/<dataset>/<route>/<model>.yaml`
* **Splits:** `splits/<task>/<dataset>/` (e.g., `slide_split.csv`, `fold_0.csv`, or dataset-specific manifests)

Refer to [`configs/README.md`](configs/README.md) and [`splits/README.md`](splits/README.md) for complete inventories and protocol details.

## License and acknowledgements

The benchmark code in this repository is released under the [MIT License](LICENSE) `[or Apache-2.0 — update to match your LICENSE file]`. Pretrained weights and upstream model adapters remain subject to the licenses and terms of use of their respective authors (`UNI2-h`, `GigaTIME`, `MIPHEI-ViT`, `HEMIT`, `ROSIE`, `HistoPlexer`, and `ASlide`). All evaluated datasets are publicly available cohort studies; please cite the original dataset publications alongside this benchmark when using the splits.

## Citation

If you find this benchmark, codebase, or split protocol useful in your research, please cite our preprint:

```bibtex
@article{zhao2026hemif,
  title   = {[Exact Paper Title]},
  author  = {Zhao, Fuxian and [Co-authors]},
  journal = {arXiv preprint arXiv:2611.19905},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.19905}
}
```
