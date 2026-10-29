# Configurations

YAML files define the dataset registry, extraction protocol, feature experiments, model settings, and downstream evaluation runs.

## Downstream runs

Each downstream run has its own YAML:

```text
configs/downstream/<task>/<dataset>/<route>/<model>.yaml
```

`encoder` evaluates frozen encoder features. `output` evaluates features from the model's generated mIF output when that route is available. Dataset-specific settings select the loader, input geometry, head, training options, postprocessing, and metrics.

Run one configuration or all model routes for a task from the repository root:

```bash
python -m training.train --config configs/downstream/prognosis/tcga_coad/encoder/uni2.yaml
python -m training.pipeline --task prognosis
```

Supplementary single-cell configurations for PCA features and the H-optimus-0 encoder are stored under `single_cell_classification_pca/` and `single_cell_classification_hoptimus/`.

## Dataset paths

`datasets.yaml` maps dataset names to local image roots, frozen split files, and table columns. Set `HE_MIF_DATASETS_ROOT` for a common data root or override a dataset with `PREPATH_DATASET_<NAME>_DIR`. Set `HE_MIF_CELL_NICHES_RAW` only when rebuilding paired Cell-niches patches from the source FOVs.

## Feature extraction

`experiments/*.yaml` selects a model, dataset, layers, and feature output location. `extraction/protocol.yaml` defines the shared layer names, patch geometry, and storage layout.

```bash
python -m feature_extraction.extract \
  --config configs/experiments/abmil_tcga_coad_rosie_40x_s128.yaml
```

Feature extraction writes to the configured `Pre_Features/` location. See the repository [`README.md`](../README.md) for setup and end-to-end reproduction steps.
