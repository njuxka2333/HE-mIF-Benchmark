"""Load benchmark experiment YAML configs."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from data.loader import DatasetConfig, get_dataset_config
from utils.paths import CONFIG_DIR, PREPATH_ROOT, PRE_FEATURES_ROOT, relocate_lab_path, resolve_project_path


def _default_layer_feat_dirs(
    feat_dir: Path,
    layers: List[str],
    model: Optional[str] = None,
) -> Dict[str, Path]:
    if model:
        try:
            from feature_extraction.protocol import protocol_layer_feat_dirs

            return protocol_layer_feat_dirs(feat_dir, model, layers)
        except KeyError:
            pass
    mapping: Dict[str, Path] = {}
    for index, layer in enumerate(layers):
        if index == 0:
            mapping[layer] = Path(feat_dir)
        else:
            mapping[layer] = Path(f"{feat_dir}__{layer}")
    return mapping


def _resolve_layer_feat_dirs(
    feat_dir: Path,
    layers: List[str],
    split_layer_feat_dirs: bool = False,
    layer_feat_dirs: Optional[Dict[str, Path]] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Path]]:
    if not split_layer_feat_dirs or len(layers) <= 1:
        return None
    if layer_feat_dirs:
        missing = [layer for layer in layers if layer not in layer_feat_dirs]
        if missing:
            raise KeyError(
                f"run.layer_feat_dirs missing entry for target_features layer(s) {missing}"
            )
        return {layer: Path(layer_feat_dirs[layer]) for layer in layers}
    return _default_layer_feat_dirs(feat_dir, layers, model=model)


EXPERIMENTS_DIR = CONFIG_DIR / "experiments"


def parse_target_features(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.replace("-> GAP", "").replace("->GAP", "")
        items = text.split(",")
    else:
        return []

    def norm(token: Any) -> str:
        return re.sub(r"\s+", "", str(token)).strip()

    return [norm(x) for x in items if norm(x)]


@dataclass
class ExtractionTask:
    """Task context passed to feature extractors."""

    head: str
    model: str
    target_features: List[str]
    use_gap: bool = True
    dense_interface: Optional[str] = None


@dataclass
class OfficialSamplingSettings:
    """HookNet-TLS sampling over the official merged XML masks.

    When ``fixed_grid_root`` is set, training reads pre-extracted PNG patches from
    ``{fixed_grid_root}/{split}/``. Default ``fixed_grid_sampler_mode=index`` uses
    PathoCell PFMB tiling over each canonical 1024 px cell
    (``FixedGridIndexSampler``). WSI-while-train sampling is not supported.
    """

    start_epoch: int = 0
    epochs: int = 200
    steps: int = 250
    sampler_batch_size: int = 24
    seed: int = 42
    fast: bool = True
    progress_every: int = 1000
    wsi_cache_size: int = 4
    prefetch_batches: int = 2
    mask_cache_size: int = 65536
    fixed_grid_root: Optional[Path] = None
    fixed_grid_patch_weight: str = "uniform"
    fixed_grid_min_selected_class_pixels: int = 1
    fixed_grid_max_resample_attempts: int = 32
    fixed_grid_sampler_mode: str = "index"


@dataclass
class ExperimentConfig:
    """Single benchmark experiment."""

    id: str
    head: str
    dataset: str
    model: str
    target_mpp: Optional[float]
    target_patch_size: int
    target_features: List[str]
    use_gap: bool
    save_masks: bool
    dataset_config: DatasetConfig
    split_filter: Optional[str]
    feat_dir: Path
    batch_size: Optional[int]
    feature_extract_batch_size: Optional[int]
    use_benchmark_config: bool
    benchmark_config_path: Optional[Path]
    fallback_mpp: float
    output_layout: str
    feature_storage: str
    grid_stride: int
    feature_dtype: str
    mask_dtype: str
    dense_interface: Optional[str] = None
    input_format: str = "manifest"
    wds_root: Optional[Path] = None
    extraction_mode: str = "grid"
    backend: str = "openslide"
    cpu_threads: int = 4
    iterator_cpus: int = 4
    progress_every: int = 500
    coords_manifest: Optional[Path] = None
    official_sampling: Optional[OfficialSamplingSettings] = None
    pathocell_tiling: str = "pfmb"
    split_layer_feat_dirs: bool = False
    layer_feat_dirs: Optional[Dict[str, Path]] = None
    feat_dir_base: Optional[Path] = None

    @property
    def use_mpp(self) -> bool:
        return self.target_mpp is not None and self.target_mpp > 0

    @property
    def gpu_extract_batch_size(self) -> int:
        """GPU micro-batch for feature extraction (grid or official merged)."""
        if self.feature_extract_batch_size is not None:
            return int(self.feature_extract_batch_size)
        if self.batch_size is not None:
            return int(self.batch_size)
        return 32

    @property
    def is_wsi(self) -> bool:
        return bool(self.dataset_config.is_wsi)

    @property
    def num_workers(self) -> int:
        return int(self.iterator_cpus)

    def extraction_task(self) -> ExtractionTask:
        return ExtractionTask(
            head=self.head,
            model=self.model,
            target_features=list(self.target_features),
            use_gap=self.use_gap,
            dense_interface=self.dense_interface,
        )


def _resolve_path(value: Any, base: Path) -> Path:
    path = relocate_lab_path(value)
    if path.is_absolute():
        return path
    return relocate_lab_path(resolve_project_path(path, base))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_file():
        alt = EXPERIMENTS_DIR / path.name
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(f"Experiment config not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    exp = raw.get("experiment") or {}
    masks = raw.get("masks") or {}
    data = raw.get("data") or {}
    run = raw.get("run") or {}

    exp_id = str(exp.get("id") or path.stem)
    dataset_name = str(exp["dataset"])
    dataset_config = get_dataset_config(dataset_name)
    default_feat = PRE_FEATURES_ROOT / exp_id
    feat_dir = _resolve_path(run.get("feat_dir", default_feat), PREPATH_ROOT)

    bench_path = run.get("benchmark_config_path")
    if bench_path:
        benchmark_config_path = _resolve_path(bench_path, PREPATH_ROOT)
    else:
        benchmark_config_path = CONFIG_DIR / "benchmark_models.yaml"

    patch_size = int(exp.get("target_patch_size", -1))
    target_mpp = exp.get("target_mpp")
    if target_mpp is not None:
        target_mpp = float(target_mpp)

    model_name = str(exp["model"])
    target_features = parse_target_features(exp.get("target_features"))

    dense_interface_raw = exp.get("dense_interface")
    dense_interface = str(dense_interface_raw) if dense_interface_raw is not None else None
    if dense_interface is not None:
        try:
            from heads.interfaces import get_default_interface, get_legacy_interface
        except ImportError as exc:
            raise ValueError(
                f"{path}: cannot import unetr interface helpers: {exc}"
            ) from exc
        try:
            if dense_interface in {"", "default", None}:
                interface = get_default_interface(model_name)
            else:
                interface = get_legacy_interface(model_name, dense_interface)
        except Exception as exc:
            raise ValueError(
                f"{path}: unsupported dense_interface={dense_interface!r}; {exc}"
            ) from exc
        canonical = list(interface["target_features"])
        if target_features and target_features != canonical:
            raise ValueError(
                f"{path}: dense interface for model {exp['model']!r} requires ordered features "
                f"{canonical}, got {target_features}"
            )
        target_features = canonical

    head = str(exp["head"])
    use_gap_raw = exp.get("use_gap")
    if use_gap_raw is None:
        use_gap = head in {"linear_probe", "abmil"}
    else:
        use_gap = bool(use_gap_raw)

    split_filter = data.get("split_filter") or data.get("split")

    feature_storage = str(run.get("feature_storage", "auto"))
    if feature_storage == "auto":
        feature_storage = "patch_h5" if head == "unetr" else "legacy_pt"

    extraction_mode = str(run.get("extraction_mode", "grid"))
    raw_official = run.get("official_sampling")
    official_sampling = None
    if raw_official:
        if extraction_mode == "grid":
            extraction_mode = "official_sampling"
        if raw_official.get("spacing") is not None:
            warnings.warn(
                f"{path}: run.official_sampling.spacing is ignored; using experiment.target_mpp for WSI patch spacing.",
                stacklevel=1,
            )
        strict = bool(raw_official.get("strict", False))
        fast = bool(raw_official.get("fast", not strict))
        fixed_grid_root_raw = raw_official.get("fixed_grid_root")
        fixed_grid_root = (
            _resolve_path(fixed_grid_root_raw, PREPATH_ROOT) if fixed_grid_root_raw else None
        )
        patch_weight = str(
            raw_official.get("fixed_grid_patch_weight", "uniform")
        ).strip().lower()
        if patch_weight not in {"uniform", "pixel"}:
            raise ValueError(
                f"{path}: official_sampling.fixed_grid_patch_weight must be 'uniform' or 'pixel' (got {patch_weight!r})"
            )
        sampler_mode = str(
            raw_official.get("fixed_grid_sampler_mode", "index")
        ).strip().lower()
        if sampler_mode not in {"index", "label_balanced"}:
            raise ValueError(
                f"{path}: official_sampling.fixed_grid_sampler_mode must be 'index' or 'label_balanced' (got {sampler_mode!r})"
            )
        official_sampling = OfficialSamplingSettings(
            start_epoch=int(raw_official.get("start_epoch", 0)),
            epochs=int(raw_official.get("epochs", 200)),
            steps=int(raw_official.get("steps", 250)),
            sampler_batch_size=int(raw_official.get("sampler_batch_size", 24)),
            seed=int(raw_official.get("seed", 42)),
            fast=fast,
            progress_every=int(raw_official.get("progress_every", 1000)),
            wsi_cache_size=int(raw_official.get("wsi_cache_size", 4)),
            prefetch_batches=int(raw_official.get("prefetch_batches", 2)),
            mask_cache_size=int(raw_official.get("mask_cache_size", 65536)),
            fixed_grid_root=fixed_grid_root,
            fixed_grid_patch_weight=patch_weight,
            fixed_grid_min_selected_class_pixels=max(
                1, int(raw_official.get("fixed_grid_min_selected_class_pixels", 1))
            ),
            fixed_grid_max_resample_attempts=max(
                1, int(raw_official.get("fixed_grid_max_resample_attempts", 32))
            ),
            fixed_grid_sampler_mode=sampler_mode,
        )
        if fixed_grid_root is not None:
            extraction_mode = "fixed_grid"
            if "feature_storage" not in run or str(run.get("feature_storage", "auto")) == "auto":
                feature_storage = "none"
            training_block = (
                raw.get("training") if isinstance(raw.get("training"), dict) else {}
            )
            data_block = raw.get("data") if isinstance(raw.get("data"), dict) else {}
            valid_proto = str(
                (raw.get("training") or {}).get("online_valid_protocol")
                or training_block.get("online_valid_protocol")
                or "sampled"
            ).strip().lower()
            test_proto = str(
                (raw.get("data") or {}).get("test_eval_protocol")
                or data_block.get("test_eval_protocol")
                or "fixed_sampled"
            ).strip().lower()
            if valid_proto not in {
                "grid",
                "fixed_sampled",
                "fast",
                "sampled",
                "full_grid",
                "online_sampler",
            }:
                raise ValueError(
                    f"{path}: with official_sampling.fixed_grid_root, training.online_valid_protocol must be 'sampled' or 'full_grid' (got {valid_proto!r})"
                )
            if test_proto in {"sliding_window", "grid"}:
                raise ValueError(
                    f"{path}: with official_sampling.fixed_grid_root, data.test_eval_protocol must be 'fixed_sampled' or 'full_grid' (got {test_proto!r})"
                )
    elif extraction_mode == "official_sampling":
        raise ValueError(f"{path}: extraction_mode=official_sampling requires run.official_sampling")

    if dataset_name == "hooknet_tls":
        if official_sampling is None or official_sampling.fixed_grid_root is None:
            raise ValueError(
                f"{path}: TLS requires official_sampling.fixed_grid_root "
                "(pre-cropped PNG grid); WSI-while-train sampling is removed"
            )

    coords_manifest_raw = run.get("coords_manifest")
    coords_manifest = (
        _resolve_path(coords_manifest_raw, PREPATH_ROOT) if coords_manifest_raw else None
    )
    wds_root_raw = run.get("wds_root")
    wds_root = _resolve_path(wds_root_raw, PREPATH_ROOT) if wds_root_raw else None

    pathocell_tiling = str(
        exp.get("pathocell_tiling") or data.get("pathocell_tiling") or "pfmb"
    ).strip().lower()

    split_layer_feat_dirs = bool(run.get("split_layer_feat_dirs", False))
    raw_layer_dirs = run.get("layer_feat_dirs")
    resolved_layer_dirs_raw = None
    if isinstance(raw_layer_dirs, dict) and raw_layer_dirs:
        resolved_layer_dirs_raw = {
            str(key): _resolve_path(value, PREPATH_ROOT)
            for key, value in raw_layer_dirs.items()
        }
    layer_feat_dirs = _resolve_layer_feat_dirs(
        feat_dir,
        target_features,
        split_layer_feat_dirs=split_layer_feat_dirs,
        layer_feat_dirs=resolved_layer_dirs_raw,
        model=model_name,
    )

    return ExperimentConfig(
        id=exp_id,
        head=head,
        dataset=dataset_name,
        model=model_name,
        target_mpp=target_mpp,
        target_patch_size=patch_size,
        target_features=target_features,
        use_gap=use_gap,
        save_masks=bool(masks.get("save", False)),
        dataset_config=dataset_config,
        split_filter=str(split_filter) if split_filter else None,
        feat_dir=feat_dir,
        batch_size=run.get("batch_size"),
        feature_extract_batch_size=run.get("feature_extract_batch_size"),
        use_benchmark_config=bool(run.get("use_benchmark_config", True)),
        benchmark_config_path=benchmark_config_path,
        fallback_mpp=float(run.get("fallback_mpp", 0.25)),
        output_layout=str(run.get("output_layout", "prepath_legacy")),
        feature_storage=feature_storage,
        grid_stride=int(run.get("grid_stride", 2)),
        feature_dtype=str(run.get("feature_dtype", "float16")),
        mask_dtype=str(run.get("mask_dtype", "uint8")),
        dense_interface=dense_interface,
        input_format=str(run.get("input_format", "manifest")),
        wds_root=wds_root,
        extraction_mode=extraction_mode,
        backend=str(run.get("backend", "openslide")),
        cpu_threads=int(run.get("cpu_threads", 4)),
        iterator_cpus=int(run.get("num_workers", run.get("iterator_cpus", 4))),
        progress_every=int(run.get("progress_every", 500)),
        coords_manifest=coords_manifest,
        official_sampling=official_sampling,
        pathocell_tiling=pathocell_tiling,
        split_layer_feat_dirs=split_layer_feat_dirs,
        layer_feat_dirs=layer_feat_dirs,
        feat_dir_base=feat_dir,
    )


def list_experiment_configs() -> List[Path]:
    return sorted(EXPERIMENTS_DIR.glob("*.yaml"))
