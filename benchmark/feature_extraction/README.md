# Feature extraction

Extract frozen model features with an experiment configuration:

```bash
python -m feature_extraction.extract \
  --config configs/experiments/abmil_tcga_coad_rosie_40x_s128.yaml
```

The experiment YAML selects the dataset, model, layers, patch geometry, and output location. `configs/extraction/protocol.yaml` defines the shared layer naming and feature storage layout. Use `--dataset`, `--model`, and `--layers` to select a run from the command line when an experiment configuration is not supplied.

Feature bags are written to the configured `Pre_Features/` directory and are consumed by offline downstream tasks. Prepare local data and set dataset paths before extraction; see the repository [`README.md`](../../README.md).
