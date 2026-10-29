# Dataset preparation

Prepare the local image and patch assets used by the benchmark through the dataset entry point:

```bash
python -m preprocessing.build --dataset hooknet_tls
python -m preprocessing.build --dataset lizard
python -m preprocessing.build --dataset immunocto
```

Use the frozen assignments and catalogs in `splits/` for reproduction. Split CSVs are inputs to the benchmark protocol; rebuilding them changes the evaluation and is not part of a standard rerun. See the repository [`README.md`](../../README.md) for dataset setup and task commands.
