# HE–mIF Benchmark

Six pathology models, one extraction protocol, five downstream tasks.
YAML under `configs/` is the recipe. `splits/` is the frozen assignment of
cases, cells, and tiles. `benchmark/` is the code. Licensed under MIT
(`LICENSE`).

Official entrance: `python -m training.pipeline` with
`PYTHONPATH` pointing at `benchmark/` (and ASlide when needed).
One YAML is `python -m training.train --config …`. Feature bags are
`python -m feature_extraction.extract`. Local patches are
`python -m preprocessing.build --dataset …`.
Do not launch a paper rerun from `Evaluation/` — that folder is scratch
written by a local run.

## What this repo contains

| Path | Role |
|------|------|
| `benchmark/` | Core code (training, models, data, metrics, …) |
| `configs/` | YAML recipes (datasets, models, downstream, extraction) |
| `test/` | Pytest suite |
| `splits/` | Frozen train/valid/test tables |

Not in git (created locally when you run; do **not** `git add`):

| Path | Role |
|------|------|
| `checkpoints/` | Weights that are not loaded from Hugging Face |
| `Pre_Features/` | Extracted feature bags |
| `results/` | Frozen paper tables (local snapshot; not required to rerun) |
| `Evaluation/` | Scratch train dirs and example tiles for this clone |

## Setup

Python 3.10+ with a CUDA PyTorch wheel. From this directory:

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # then set HF_TOKEN (GigaTIME, UNI2-h, and H-optimus-0 are gated)
```

Before a run, clone the upstream model repos **into this directory** (UNI2
has no code checkout; it loads from Hugging Face):

```bash
git clone https://gitlab.com/enable-medicine-public/rosie.git rosie
mkdir -p HEMIT
git clone https://github.com/BianChang/Pix2pix_DualBranch.git HEMIT/Pix2pix_DualBranch
git clone https://github.com/prov-gigatime/GigaTIME.git GigaTIME
git clone https://github.com/Sanofi-Public/MIPHEI-ViT.git MIPHEI-ViT
git clone https://github.com/ratschlab/HistoPlexer.git HistoPlexer
git clone https://github.com/MrPeterJin/ASlide ASlide
```

Then:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/benchmark:$PWD/ASlide${PYTHONPATH:+:$PYTHONPATH}"
export HE_MIF_DATASETS_ROOT=/path/to/datasets
# Optional override; default is $PWD/checkpoints
# export HE_MIF_CKPT_DIR=/path/to/checkpoints
# Immunocto is not under the datasets root on the original machine:
# export PREPATH_DATASET_IMMUNOCTO_DIR=/path/to/ImmunoctoV1
# Cell-niches raw HE/mIF FOVs (needed only to rebuild paired patches):
# export HE_MIF_CELL_NICHES_RAW=/path/to/Cell-niches-data
```

`configs/datasets.yaml` still lists this lab's absolute paths. Setting
`HE_MIF_DATASETS_ROOT` remaps anything under `/data3/houjunlin/fxx/datasets`.
Override one dataset with `PREPATH_DATASET_<NAME>_DIR` (name uppercased).
`HE_MIF_CELL_NICHES_RAW` is required off this machine to rebuild Cell-niches
tiles. Leftover YAML paths that still name this lab's clone are remapped to
the current repository root.

## Checkpoints

All local weights live in **this repo's** `checkpoints/` (override with
`$HE_MIF_CKPT_DIR`). Do not keep a second copy outside the repo.

```bash
mkdir -p checkpoints
```

A file already in `checkpoints/` wins. GigaTIME / UNI2 / MIPHEI still fall
back to Hub when that file is missing.

| Model | Put this file in `checkpoints/` | How to get it |
|-------|----------------------------------|---------------|
| UNI2-h | `uni2_pytorch_model.bin` | Hugging Face `MahmoodLab/UNI2-h` (`pytorch_model.bin`). Gated; needs `HF_TOKEN`. |
| GigaTIME | `gigatime.pth` | Hugging Face `prov-gigatime/GigaTIME` (`model.pth`). Gated; needs `HF_TOKEN`. |
| MIPHEI-ViT | `mipheivit.safetensors` | GitHub release `v1.0.0` `model.safetensors`. Encoder `bioptimus/H-optimus-0` is gated on Hub and loaded by the MIPHEI code. |
| HEMIT | `hemit_v1.pth` | Official Google Drive id `1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd`. |
| ROSIE | `rosie_best_model_single.pth` | Request `best_model_single.pth` from [ericwu09/ROSIE](https://huggingface.co/ericwu09/ROSIE) and rename it. |
| HistoPlexer | `histoplexer.pt` | After cloning HistoPlexer: `cp HistoPlexer/demo/model/pytorch_model.pt checkpoints/histoplexer.pt`. |

```bash
# UNI2-h
huggingface-cli download MahmoodLab/UNI2-h pytorch_model.bin --local-dir /tmp/uni2
mv /tmp/uni2/pytorch_model.bin checkpoints/uni2_pytorch_model.bin

# GigaTIME
huggingface-cli download prov-gigatime/GigaTIME model.pth --local-dir /tmp/gigatime
mv /tmp/gigatime/model.pth checkpoints/gigatime.pth

# MIPHEI-ViT generator head
wget -O checkpoints/mipheivit.safetensors \
  https://github.com/Sanofi-Public/MIPHEI-ViT/releases/download/v1.0.0/model.safetensors

# HEMIT
pip install gdown
gdown --id 1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd -O checkpoints/hemit_v1.pth

# ROSIE (after HF access is granted)
huggingface-cli download ericwu09/ROSIE best_model_single.pth --local-dir /tmp/rosie
mv /tmp/rosie/best_model_single.pth checkpoints/rosie_best_model_single.pth

# HistoPlexer
cp HistoPlexer/demo/model/pytorch_model.pt checkpoints/histoplexer.pt
```

## Reproduce a reported table

From the repo root. Build local patch files from the frozen splits, extract
features where the task is offline, then train.

```bash
python -m preprocessing.build --dataset tcga_coad
python -m preprocessing.build --dataset tcga_luad
python -m preprocessing.build --dataset hooknet_tls
python -m preprocessing.build --dataset cell_niches
python -m preprocessing.build --dataset lizard
python -m preprocessing.build --dataset immunocto

python -m feature_extraction.extract --dataset immunocto --model uni2 --layers all
python -m training.pipeline --task single_cell_classification
python -m training.pipeline --all
```

`--models uni2` and `--route encoder` narrow one arm. `--test-only` evaluates
an existing checkpoint. `training.postprocess` is called by the pipeline; it
is not a separate entrance. PathoCell and NuCLS use the released patch files
(`preprocessing.build` is a no-op for those names).

WSI extract uses `--num-workers 0`.

A train command writes under `Evaluation/` (or the `run.output_dir` in that
YAML). Copy a finished run into `results/` only when you intend to replace a
published arm. Do not `git add` `results/`, `Pre_Features/`, or `checkpoints/`.

The MIL / UNETR / linear-probe **training loops** themselves are archived
CPython 3.10/3.11 bytecode under `benchmark/training/_orig_pyc/`, with thin
Python wrappers that apply the paper patches. Source-level heads, metrics,
routing, and YAML live next to those files. Use Python 3.10 or 3.11 to rerun
the published numbers.

### Virtual staining (Cell-niches)

Protocol id is `cell_niches_shared`. Headline: CD3 Spearman + Otsu-AUPRC on
the **8 µm** grid after a 4 µm mean-pool cache, 477 patches / 20 FOVs.

```bash
python -m training.pipeline --task virtual_staining --significance
```

Equivalent one-model call:

```bash
python -m training.train --config configs/downstream/virtual_staining/cell_niches/output/rosie.yaml
```

Do not recut `splits/virtual_staining/patches.csv`. Generator feeds are
paper-native. Rows are locked model order, not a Spearman ranking. Do not
publish P/R/F1 at cutoff 0.5.

The pipeline writes the official example tiles after scoring:

- virtual staining → `Evaluation/virtual_staining/example_tile/`
- TLS → `Evaluation/tls_seg/example_tile/`
- cell semantic segmentation → `Evaluation/cell_semantic_seg/pathocell/example_tile/`

## Models and layers

| Model | Encoder | Virtual staining |
|-------|---------|------------------|
| UNI2 | `z4` | — |
| MIPHEI-ViT | `z4` | `mif` |
| HEMIT | `resnet_blocks` | `mif` |
| HistoPlexer | `bottleneck` | `mif` |
| GigaTIME | `x4_0` | `mif` |
| ROSIE | `stage3` | `mif0` |

Patch geometry for feature extraction is in
`configs/extraction/protocol.yaml`. Online tasks (TLS / PathoCell /
Lizard / Cell-niches) keep the native input size in each downstream YAML.
UNI2 has no virtual-staining layer, so there is no `output/uni2.yaml`.

## Tasks

| Task | Datasets | Primary metric | Pipeline name |
|------|----------|----------------|---------------|
| Virtual staining | Cell-niches | CD3 Spearman + AUPRC @ 8 µm (`cell_niches_shared`) | `virtual_staining` |
| Single-cell classification | Immunocto, NuCLS | linear-probe AUROC | `single_cell_classification` |
| Prognosis | TCGA-COAD, TCGA-LUAD | mean 5-fold C-index, `hidden_dim=256` | `prognosis` |
| TLS segmentation | HookNet-TLS | macro F1 | `tls_region_segmentation` |
| Cell semantic segmentation | PathoCell, Lizard | instance macro F1 | `cell_semantic_segmentation` |

COAD uses the Shazam 5-fold split. LUAD uses an OS-balanced 5-fold split. In
every WSI CV fold, **valid = test**. TLS tiles are `splits/tls_seg/patches.csv`.

## Tests

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/benchmark${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest test/test_configs.py test/test_pipeline.py test/test_train_router.py test/test_example_tiles.py test/test_shared_vstain_protocol.py test/test_vstain_significance.py test/test_vstain_metrics.py test/test_loader_tables.py
```

## Layout

```
benchmark/training/     load / train entry (pipeline, loops, YAML, example tiles)
benchmark/training/_orig_pyc/  archived training-loop bytecode (Python 3.10/3.11)
benchmark/heads/        MIL / UNETR / cellseg / mIF decoders
benchmark/metrics/      classification, survival, segmentation, vstain, clinical stats
benchmark/losses/
benchmark/data/         WSI/FOV loaders + extracted feature bags
benchmark/mif/          generated-mIF trainers + Cell-niches scorer
benchmark/models/       wrappers + registry/schema; checkpoint I/O in utils/checkpoints.py
configs/extraction/protocol.yaml
configs/downstream/
  linear_probe/{immunocto,nucls}/{encoder,output}/{model}.yaml
  prognosis/{tcga_coad,tcga_luad}/{encoder,output}/{model}.yaml
  tls/{encoder,output}/{model}.yaml
  cellseg/{pathocell,lizard}/{encoder,output}/{model}.yaml
  virtual_staining/cell_niches/output/{model}.yaml
splits/          frozen case / cell / tile lists
test/            pytest
```

Feature bags: Immunocto / NuCLS are `linear_probe`; TCGA COAD / LUAD are
`wsi_cv` (`valid` hardlinked to `test`); TLS / PathoCell / Lizard /
Cell-niches are online (no Pre_Features bags).
