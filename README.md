# HE–mIF Benchmark

This repository reproduces the HE-to-multiplex immunofluorescence benchmark across six pathology models and five downstream tasks. The published data assignments are versioned in `splits/`; experiment recipes are in `configs/`; executable pipelines are in `benchmark/`.

## Tasks and datasets

| Task | Dataset(s) | Main evaluation |
|---|---|---|
| Virtual staining | Cell-niches | CD3 Spearman correlation and AUPRC |
| Single-cell classification | Immunocto, NuCLS | Linear-probe AUROC |
| Prognosis | TCGA-COAD, TCGA-LUAD | Five-fold C-index |
| TLS region segmentation | HookNet-TLS | Macro F1 |
| Cell semantic segmentation | PathoCell, Lizard | Instance-level macro F1 |

## Requirements and setup

Use Python 3.10 or 3.11 and install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Several model adapters use their upstream source repositories. Clone the required repositories at the project root; UNI2 loads through Hugging Face and does not require a source checkout.

```bash
git clone https://gitlab.com/enable-medicine-public/rosie.git rosie
mkdir -p HEMIT
git clone https://github.com/BianChang/Pix2pix_DualBranch.git HEMIT/Pix2pix_DualBranch
git clone https://github.com/prov-gigatime/GigaTIME.git GigaTIME
git clone https://github.com/Sanofi-Public/MIPHEI-ViT.git MIPHEI-ViT
git clone https://github.com/ratschlab/HistoPlexer.git HistoPlexer
git clone https://github.com/MrPeterJin/ASlide ASlide
```

Set the benchmark package and upstream ASlide on `PYTHONPATH`, then configure local dataset locations:

```bash
export PYTHONPATH="$PWD/benchmark:$PWD/ASlide${PYTHONPATH:+:$PYTHONPATH}"
export HE_MIF_DATASETS_ROOT=/path/to/datasets
# Set individually when a dataset is stored outside HE_MIF_DATASETS_ROOT:
# export PREPATH_DATASET_IMMUNOCTO_DIR=/path/to/ImmunoctoV1
# Cell-niches source FOVs are needed only to rebuild paired patches:
# export HE_MIF_CELL_NICHES_RAW=/path/to/Cell-niches-data
```

`HE_MIF_DATASETS_ROOT` contains the dataset folders named in `configs/datasets.yaml`. A dataset-specific `PREPATH_DATASET_<NAME>_DIR` variable overrides its folder. Place local model weights in `checkpoints/`, or set `HE_MIF_CKPT_DIR` to use another directory. UNI2-h, GigaTIME, and the H-optimus-0 backbone require access to their gated Hugging Face repositories. Authenticate with Hugging Face and set `HF_TOKEN` before loading them.

| Model | Local checkpoint name | Source |
|---|---|---|
| UNI2-h | `uni2_pytorch_model.bin` | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| GigaTIME | `gigatime.pth` | [prov-gigatime/GigaTIME](https://huggingface.co/prov-gigatime/GigaTIME) |
| MIPHEI-ViT | `mipheivit.safetensors` | [MIPHEI-ViT v1.0.0 release](https://github.com/Sanofi-Public/MIPHEI-ViT/releases/tag/v1.0.0) |
| HEMIT | `hemit_v1.pth` | Official Google Drive file `1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd` |
| ROSIE | `rosie_best_model_single.pth` | [ericwu09/ROSIE](https://huggingface.co/ericwu09/ROSIE) |
| HistoPlexer | `histoplexer.pt` | `HistoPlexer/demo/model/pytorch_model.pt` in the upstream repository |

## Prepare data

The repository includes split tables and tile catalogs, but not whole-slide images, image patches, or extracted features. Set dataset paths before preparing the required local assets. For example:

```bash
python -m preprocessing.build --dataset hooknet_tls
python -m preprocessing.build --dataset cell_niches
python -m preprocessing.build --dataset tcga_coad
```

Run only the preparation commands needed for the task and dataset. The corresponding dataset README under `splits/` documents the frozen catalog and preprocessing protocol.

## Run the benchmark

Run one configured model or a whole task from the repository root:

```bash
python -m training.train \
  --config configs/downstream/cell_semantic_segmentation/pathocell/encoder/hemit.yaml
python -m training.pipeline --task single_cell_classification
python -m training.pipeline --task prognosis --models uni2
python -m training.pipeline --task virtual_staining --significance
```

`training.pipeline --all` runs the reported task suite. Add `--route encoder` or `--route output` to select a feature route, and `--test-only` to evaluate existing checkpoints. Feature extraction for offline tasks uses the experiment recipe and shared extraction protocol:

```bash
python -m feature_extraction.extract \
  --config configs/experiments/abmil_tcga_coad_rosie_40x_s128.yaml
```

Training and evaluation outputs are written to the configured output directory (normally `Evaluation/`). Extracted feature bags are stored under `Pre_Features/`. These outputs are local run artifacts and are not part of the frozen split tables.

## Configuration and split layout

Downstream configurations follow `configs/downstream/<task>/<dataset>/<route>/<model>.yaml`. Split files follow `splits/<task>/<dataset>/`, with names such as `slide_split.csv`, `fold_0.csv`, or a dataset-specific manifest name where the unit is not a slide. See [`configs/README.md`](configs/README.md) and [`splits/README.md`](splits/README.md) for the inventories and run instructions.
