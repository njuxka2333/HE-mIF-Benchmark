# Training and evaluation

Run a single model configuration with `training.train`, or run and compare the configurations for a task with `training.pipeline`:

```bash
python -m training.train --config configs/downstream/prognosis/tcga_coad/encoder/uni2.yaml
python -m training.pipeline --task prognosis
python -m training.pipeline --task single_cell_classification --models uni2 --route encoder
python -m training.pipeline --task virtual_staining --significance
```

Tasks are `virtual_staining`, `cell_semantic_segmentation`, `tls_region_segmentation`, `single_cell_classification`, and `prognosis`. `--all` runs the reported task suite. `--test-only` evaluates existing checkpoints; `--dry-run` lists the configurations selected for a task.

Downstream YAMLs control dataset, route, model, training, and metric settings. Set dataset locations as described in the repository [`README.md`](../../README.md). Results are written to each run's configured output directory.
