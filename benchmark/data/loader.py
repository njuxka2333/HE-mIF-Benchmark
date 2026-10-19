"""Load dataset configs and split CSV manifests."""

from __future__ import annotations

import csv
import gzip
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, TextIO, Union

import yaml

from utils.paths import CONFIG_DIR, PREPATH_ROOT, REPO_ROOT, SPLITS_ROOT, relocate_dataset_path

_CONFIG_PATH = CONFIG_DIR / "datasets.yaml"


def resolve_table_path(path: Path | None) -> Path | None:
    """Prefer the plain CSV; fall back to ``.csv.gz`` for GitHub-sized tables."""
    if path is None:
        return None
    if path.is_file():
        return path
    gz = Path(str(path) + ".gz")
    if gz.is_file():
        return gz
    return path


@contextmanager
def open_csv_table(path: Path, encoding: str = "utf-8-sig") -> Iterator[TextIO]:
    resolved = resolve_table_path(path)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(path)
    if resolved.name.endswith(".gz"):
        with gzip.open(resolved, "rt", newline="", encoding=encoding) as handle:
            yield handle
    else:
        with resolved.open(newline="", encoding=encoding) as handle:
            yield handle


class SplitLevel(str, Enum):
    CASE = "case"
    SLIDE = "slide"


@dataclass
class DatasetColumns:
    case_id: str = "case_id"
    sample_id: str = "sample_id"
    split: str = "split"
    label: str = "label"
    image: str = "image_path"
    mask: str = "mask_path"


@dataclass
class DatasetConfig:
    name: str
    dataset_dir: Path
    mask_dir: Optional[Path]
    split_path: Optional[Path]
    split_level: SplitLevel
    label_dict_path: Optional[Path]
    mode: str
    _label_dict_cache: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    columns: DatasetColumns = field(default_factory=DatasetColumns)
    wsi_subdir: str = ""
    annotations_subdir: str = ""
    tissue_mask_subdir: str = "tissue_masks"
    coords_subdir: str = "patches"
    slide_ext: str = ".svs"
    glob: str = "*.hdf"
    wsi_dirs: List[Path] = field(default_factory=list)
    split_glob: Optional[str] = None
    split_format: Optional[str] = None
    manifest_format: Optional[str] = None
    patient_split_path: Optional[Path] = None
    excluded_paths_path: Optional[Path] = None
    hdf_subdir: str = ""
    native_mpp: Optional[float] = None
    metadata_path: Optional[Path] = None
    recursive_wsi: bool = False
    discover_from_masks: bool = False
    mask_suffix: str = "_tls_mask.tif"
    row_filters: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_wsi(self) -> bool:
        return self.mode == "wsi"

    @property
    def is_patch_hdf(self) -> bool:
        return self.mode == "patch_hdf"

    def wsi_dir(self) -> Path:
        return self.dataset_dir / self.wsi_subdir if self.wsi_subdir else self.dataset_dir

    def coords_dir(self) -> Path:
        return self.dataset_dir / self.coords_subdir

    def tissue_mask_dir(self) -> Path:
        return self.dataset_dir / self.tissue_mask_subdir

    def annotations_dir(self) -> Path | None:
        if not self.annotations_subdir:
            return None
        return self.dataset_dir / self.annotations_subdir

    def hdf_dir(self) -> Path:
        return self.dataset_dir / self.hdf_subdir if self.hdf_subdir else self.dataset_dir

    @property
    def label_dict(self) -> Dict[str, Any]:
        if self._label_dict_cache is None:
            self._label_dict_cache = _load_label_dict(self.label_dict_path, self.dataset_dir)
        return self._label_dict_cache

    def wsi_search_roots(self) -> List[Path]:
        if self.wsi_dirs:
            return list(self.wsi_dirs)
        return [self.wsi_dir()]


@dataclass
class SplitSample:
    """One row from a split manifest (patch tile or slide)."""

    sample_id: str
    case_id: str
    split: str
    label: str
    image_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    extra: Dict[str, str] = field(default_factory=dict)


def _resolve_path(value: str | Path | None, base: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    posix = path.as_posix()
    if posix.startswith("splits/"):
        return SPLITS_ROOT / path.relative_to("splits")
    if posix.startswith("configs/"):
        return CONFIG_DIR / path.relative_to("configs")
    if posix.startswith("../splits/"):
        return PREPATH_ROOT / path
    # Avoid Path.resolve() here: it can block on slow or unavailable NFS mounts.
    return base / path


def _dataset_dir_env_key(name: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in name.upper())
    return f"PREPATH_DATASET_{normalized}_DIR"


def _resolve_path_list(values: Optional[Iterable[Union[str, Path]]], base: Path) -> List[Path]:
    if not values:
        return []
    resolved: List[Path] = []
    for value in values:
        path = _resolve_path(value, base)
        if path is not None:
            resolved.append(path)
    return resolved


def coerce_label_dict_keys(raw: dict | None) -> Dict[Any, Any]:
    """JSON object keys are strings; PFMB instance metrics need int class ids.

    PathoCell / Lizard ``label_dict.json`` is ``{"0": "Background", "1": "B cells", ...}``.
    ``InstanceLevelMetricAccumulator`` indexes probability channels with those keys, so
    string ids make every instance predict background and force ``f1_score_macro=0``.
    Name-keyed dicts (non-numeric keys) are left unchanged.
    """
    if not raw:
        return {}
    coerced: Dict[Any, Any] = {}
    for key, value in raw.items():
        if isinstance(key, bool):
            return dict(raw)
        if isinstance(key, int):
            coerced[key] = value
            continue
        if isinstance(key, str) and key.isdigit():
            coerced[int(key)] = value
            continue
        return dict(raw)
    return coerced


def _load_label_dict(raw: str | Path | dict | None, dataset_dir: Path) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return coerce_label_dict_keys(raw)
    path = Path(raw) if isinstance(raw, Path) else _resolve_path(raw, dataset_dir)
    if path and path.is_file():
        return coerce_label_dict_keys(json.loads(path.read_text(encoding="utf-8")))
    return {}


def _parse_columns(raw: Optional[dict]) -> DatasetColumns:
    raw = raw or {}
    return DatasetColumns(
        case_id=str(raw.get("case_id", "case_id")),
        sample_id=str(raw.get("sample_id", "sample_id")),
        split=str(raw.get("split", "split")),
        label=str(raw.get("label", "label")),
        image=str(raw.get("image", "image_path")),
        mask=str(raw.get("mask", "mask_path")),
    )


def _row_split_raw(row: dict, cfg: DatasetConfig, csv_path: Path | None = None) -> str:
    """Return the raw split field from a manifest row (format-aware)."""
    del csv_path
    return str(row.get(cfg.columns.split, "all"))


def _normalize_split_name(raw: str) -> str:
    value = str(raw or "all").strip().lower()
    aliases = {
        "tr": "train",
        "training": "train",
        "te": "test",
        "testing": "test",
        "val": "valid",
        "validation": "valid",
        "dev": "valid",
        "train_val_test": "all",
    }
    return aliases.get(value, value)


def _slide_id_from_wsi_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("TCGA-"):
        return stem.split(".", 1)[0]
    return stem


@lru_cache(maxsize=8)
def _load_dataset_registry_impl(config_path: str) -> Dict[str, DatasetConfig]:
    path = Path(config_path) if config_path else _CONFIG_PATH
    yaml_base = path.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    registry: Dict[str, DatasetConfig] = {}
    for name, meta in (raw.get("datasets") or {}).items():
        raw_dir = os.environ.get(_dataset_dir_env_key(name), meta["dataset_dir"])
        if Path(os.path.expandvars(os.path.expanduser(str(raw_dir)))).is_absolute():
            dataset_dir = relocate_dataset_path(raw_dir)
        elif path == _CONFIG_PATH:
            resolved = _resolve_path(raw_dir, REPO_ROOT)
            if resolved is None:
                raise ValueError(f"{name}: dataset_dir {raw_dir!r} did not resolve")
            dataset_dir = relocate_dataset_path(resolved)
        else:
            dataset_dir = relocate_dataset_path((yaml_base / raw_dir).resolve())
        mask_dir = _resolve_path(meta.get("mask_dir"), dataset_dir)
        if mask_dir is not None:
            mask_dir = relocate_dataset_path(mask_dir)
        split_raw = meta.get("split")
        split_path = _resolve_path(split_raw, dataset_dir) if split_raw else None
        label_raw = meta.get("label_dict")
        label_dict_path = _resolve_path(label_raw, dataset_dir) if label_raw else None
        registry[name] = DatasetConfig(
            name=name,
            dataset_dir=dataset_dir,
            mask_dir=mask_dir,
            split_path=split_path,
            split_level=SplitLevel(str(meta.get("split_level", "slide")).lower()),
            label_dict_path=label_dict_path,
            mode=str(meta.get("mode", "patch")),
            columns=_parse_columns(meta.get("columns")),
            wsi_subdir=str(meta.get("wsi_subdir", "")),
            annotations_subdir=str(meta.get("annotations_subdir", "")),
            tissue_mask_subdir=str(meta.get("tissue_mask_subdir", "tissue_masks")),
            coords_subdir=str(meta.get("coords_subdir", "patches")),
            slide_ext=str(meta.get("slide_ext", ".svs")),
            glob=str(meta.get("glob", "*.hdf")),
            wsi_dirs=[relocate_dataset_path(p) for p in _resolve_path_list(meta.get("wsi_dirs"), dataset_dir)],
            split_glob=str(meta.get("split_glob")) if meta.get("split_glob") else None,
            split_format=str(meta.get("split_format")) if meta.get("split_format") else None,
            manifest_format=str(meta.get("manifest_format")) if meta.get("manifest_format") else None,
            patient_split_path=_resolve_path(meta.get("patient_split"), dataset_dir)
            if meta.get("patient_split")
            else None,
            excluded_paths_path=_resolve_path(meta.get("excluded_paths"), dataset_dir)
            if meta.get("excluded_paths")
            else None,
            hdf_subdir=str(meta.get("hdf_subdir", "")),
            native_mpp=float(meta["native_mpp"]) if meta.get("native_mpp") is not None else None,
            metadata_path=_resolve_path(meta.get("metadata"), dataset_dir),
            recursive_wsi=bool(meta.get("recursive_wsi", False)),
            discover_from_masks=bool(meta.get("discover_from_masks", False)),
            mask_suffix=str(meta.get("mask_suffix", "_tls_mask.tif")),
            row_filters=dict(meta.get("row_filters") or {}),
        )
    return registry


def load_dataset_registry(config_path: Optional[str] = None) -> Dict[str, DatasetConfig]:
    key = str(Path(config_path).resolve()) if config_path else str(_CONFIG_PATH)
    return _load_dataset_registry_impl(key)


def get_dataset_config(name: str, config_path: Optional[str] = None) -> DatasetConfig:
    registry = load_dataset_registry(config_path)
    if name not in registry:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(registry)}")
    return registry[name]


def find_wsi_path(slide_id: str, cfg: DatasetConfig) -> Optional[Path]:
    """Locate a WSI file by slide id across configured search roots."""
    ext = cfg.slide_ext.lstrip(".")
    for root in cfg.wsi_search_roots():
        if not root.is_dir():
            continue
        direct = root / f"{slide_id}{cfg.slide_ext}"
        if direct.is_file():
            return direct.resolve()
        if cfg.recursive_wsi:
            for path in root.rglob(f"*{slide_id}*.{ext}"):
                if path.is_file():
                    return path.resolve()
    return None


def build_wsi_path_map(cfg: DatasetConfig) -> Dict[str, Path]:
    """Return {slide_id: absolute_wsi_path} for all slides under search roots."""
    ext = cfg.slide_ext.lstrip(".")
    result: Dict[str, Path] = {}
    for root in cfg.wsi_search_roots():
        if not root.is_dir():
            continue
        paths = root.rglob(f"*.{ext}") if cfg.recursive_wsi else root.glob(f"*{cfg.slide_ext}")
        for path in paths:
            if not path.is_file():
                continue
            slide_id = _slide_id_from_wsi_name(path)
            result.setdefault(slide_id, path.resolve())
    return result


def _resolve_media_path(raw_path: str, cfg: DatasetConfig, row: dict, kind: str) -> Optional[Path]:
    col = cfg.columns.image if kind == "image" else cfg.columns.mask
    value = raw_path or row.get(col, "")
    if not value:
        if kind == "mask" and cfg.mask_dir:
            sample_id = str(row.get(cfg.columns.sample_id, "")).strip()
            if sample_id and cfg.discover_from_masks:
                mask_name = f"{sample_id}{cfg.mask_suffix}"
                candidate = cfg.mask_dir / mask_name
                if candidate.is_file():
                    return candidate.resolve()
            image_path = _resolve_media_path(row.get(cfg.columns.image, ""), cfg, row, "image")
            if image_path:
                candidate = cfg.mask_dir / image_path.name
                if candidate.is_file():
                    return candidate.resolve()
        return None

    path = Path(value)
    if path.is_file():
        return path

    case_id = row.get(cfg.columns.case_id, "")
    sample_id = row.get(cfg.columns.sample_id, "")

    candidates: List[Path] = []
    if not path.is_absolute():
        candidates.append(cfg.dataset_dir / path)
        if path.name != str(path):
            candidates.append(cfg.dataset_dir / path.name)

    folder = cfg.dataset_dir.name
    for part in (str(path), value):
        if folder in part:
            rel = part.split(folder)[-1].lstrip("/\\")
            candidates.append(cfg.dataset_dir / rel)

    if case_id and sample_id:
        candidates.extend(
            [
                cfg.dataset_dir / case_id / "HE" / f"{sample_id}.png",
                cfg.dataset_dir / case_id / "mask" / f"{sample_id}.png",
                cfg.dataset_dir / case_id / "HE" / path.name,
                cfg.dataset_dir / case_id / "mask" / path.name,
            ]
        )

    if cfg.mask_dir and kind == "mask":
        candidates.append(cfg.mask_dir / path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_hdf_path(sample_id: str, cfg: DatasetConfig) -> Optional[Path]:
    name = sample_id if sample_id.endswith(".hdf") else f"{sample_id}.hdf"
    for candidate in (
        cfg.hdf_dir() / name,
        cfg.dataset_dir / name,
        cfg.dataset_dir / cfg.hdf_subdir / name if cfg.hdf_subdir else None,
    ):
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


def _row_to_sample(
    row: dict,
    cfg: DatasetConfig,
    csv_path: Path | None = None,
) -> Optional[SplitSample]:
    cols = cfg.columns
    case_id = str(row.get(cols.case_id, row.get(cols.sample_id, ""))).strip()
    sample_id = str(row.get(cols.sample_id, case_id)).strip()
    if not sample_id:
        return None

    image_path = _resolve_media_path(row.get(cols.image, ""), cfg, row, "image")
    mask_path = _resolve_media_path(row.get(cols.mask, ""), cfg, row, "mask")

    if cfg.is_patch_hdf and image_path is None:
        image_path = _resolve_hdf_path(sample_id, cfg)

    if cfg.mode == "patch" and image_path is None:
        return None

    if cfg.is_wsi and image_path is None:
        auto = cfg.wsi_dir() / f"{sample_id}{cfg.slide_ext}"
        if auto.is_file():
            image_path = auto.resolve()
        else:
            image_path = find_wsi_path(sample_id, cfg)

    if cfg.is_wsi and mask_path is None and cfg.mask_dir:
        mask_candidate = cfg.mask_dir / f"{sample_id}{cfg.mask_suffix}"
        if mask_candidate.is_file():
            mask_path = mask_candidate.resolve()

    if cfg.is_patch_hdf and image_path is None:
        return None

    split_value = _normalize_split_name(str(row.get(cols.split, "all")))
    label_value = str(row.get(cols.label, "")).strip()

    return SplitSample(
        sample_id=sample_id,
        case_id=case_id or sample_id,
        split=split_value,
        label=label_value,
        image_path=image_path,
        mask_path=mask_path,
        extra={k: str(v) for k, v in row.items()},
    )


def _iter_split_csv_paths(cfg: DatasetConfig) -> List[Path]:
    if cfg.split_path is not None and cfg.split_path.is_file():
        return [cfg.split_path]
    paths: List[Path] = []
    if cfg.split_path is not None and cfg.split_path.is_dir():
        paths.extend(sorted(cfg.split_path.glob("*.csv")))
    if not paths and cfg.split_glob:
        paths.extend(sorted(cfg.dataset_dir.glob(cfg.split_glob)))
    return paths


def _split_filter_matches(
    row: dict,
    cfg: DatasetConfig,
    split_filter: Optional[str],
    csv_path: Path | None = None,
) -> bool:
    if not split_filter or split_filter.lower() in {"all", "*", ""}:
        return True
    row_split = _normalize_split_name(_row_split_raw(row, cfg, csv_path))
    want = _normalize_split_name(split_filter)
    return row_split == want


def _coerce_filter_value(value: str) -> Any:
    normalized = str(value).strip().lower()
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    if normalized in {"true", "1", "yes", "y"}:
        return True
    return normalized


def _row_filters_match(row: dict, cfg: DatasetConfig) -> bool:
    for column, expected in cfg.row_filters.items():
        actual = _coerce_filter_value(row.get(column, ""))
        if isinstance(expected, bool):
            if actual is not expected:
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            allowed = {_coerce_filter_value(v) for v in expected}
            if actual not in allowed:
                return False
            continue
        if actual != _coerce_filter_value(expected):
            return False
    return True


def _load_split_csv(
    cfg: DatasetConfig,
    *,
    split_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[SplitSample]:
    csv_paths = _iter_split_csv_paths(cfg)
    if not csv_paths:
        return []

    samples: List[SplitSample] = []
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            for row in reader:
                if not _split_filter_matches(row, cfg, split_filter, csv_path):
                    continue
                if not _row_filters_match(row, cfg):
                    continue
                sample = _row_to_sample(row, cfg, csv_path)
                if sample:
                    samples.append(sample)
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
    return samples


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _relative_manifest_path(raw: str, cfg: DatasetConfig) -> str:
    value = str(raw).strip()
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(cfg.dataset_dir))
        except ValueError:
            return str(path)
    return str(path)


def _load_excluded_relative_paths(cfg: DatasetConfig) -> set[str]:
    path = resolve_table_path(cfg.excluded_paths_path)
    if path is None or not path.is_file():
        return set()
    excluded: set[str] = set()
    with open_csv_table(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("image_path") or row.get("path") or row.get("relative_path") or ""
            rel = _relative_manifest_path(raw, cfg)
            if rel:
                excluded.add(rel)
    return excluded


def _label_from_immunocto_image_id(image_id: str) -> str:
    parts = str(image_id).rsplit("_", 2)
    return parts[0] if len(parts) == 3 else str(image_id).split("_", 1)[0]


_IMMUNOCTO_CELL_INDEX = SPLITS_ROOT / "single_cell_cls" / "immunocto" / "cell_index.csv"


def _load_immunocto_index_csv(
    cfg: DatasetConfig,
    index_path: Path,
    *,
    split_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[SplitSample]:
    wanted = None
    if split_filter and split_filter.lower() not in {"all", "*", ""}:
        wanted = _normalize_split_name(split_filter)
    samples: List[SplitSample] = []
    with open_csv_table(index_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = _normalize_split_name(str(row.get("split") or "all"))
            if wanted is not None and split != wanted:
                continue
            rel = str(row["image_path"]).strip()
            samples.append(
                SplitSample(
                    sample_id=str(row["sample_id"]).strip(),
                    case_id=str(row["case_id"]).strip(),
                    split=split,
                    label=str(row["label"]).strip(),
                    image_path=cfg.dataset_dir / rel,
                    mask_path=None,
                    extra={"image_path": rel},
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def _load_immunocto_compact_manifest(
    cfg: DatasetConfig,
    *,
    split_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[SplitSample]:
    split_path = cfg.patient_split_path or cfg.split_path
    if split_path is None or not split_path.is_file():
        return []

    index_path = resolve_table_path(
        Path(os.environ.get("IMMUNOCTO_CELL_INDEX") or _IMMUNOCTO_CELL_INDEX)
    )
    if index_path is not None and index_path.is_file():
        return _load_immunocto_index_csv(
            cfg, index_path, split_filter=split_filter, max_samples=max_samples
        )

    wanted_split = None
    if split_filter and split_filter.lower() not in {"all", "*", ""}:
        wanted_split = _normalize_split_name(split_filter)

    patients: list[tuple[str, str]] = []
    with split_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = str(row.get("patient_id") or row.get("case_id") or "").strip()
            split = _normalize_split_name(str(row.get("split", "all")))
            if not patient_id:
                continue
            if wanted_split is not None and split != wanted_split:
                continue
            patients.append((patient_id, split))

    excluded = _load_excluded_relative_paths(cfg)
    samples: List[SplitSample] = []
    for patient_id, split in patients:
        he_dir = cfg.dataset_dir / patient_id / "HE"
        if not he_dir.is_dir():
            continue
        for image_path in he_dir.glob("*.png"):
            image_id = image_path.stem
            rel = str(image_path.relative_to(cfg.dataset_dir))
            if rel in excluded:
                continue
            label = _label_from_immunocto_image_id(image_id)
            samples.append(
                SplitSample(
                    sample_id=image_id,
                    case_id=patient_id,
                    split=split,
                    label=label,
                    image_path=image_path,
                    mask_path=None,
                    extra={"image_path": rel},
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def _load_metadata_samples(cfg: DatasetConfig) -> List[SplitSample]:
    meta_path = cfg.metadata_path
    if meta_path is None or not meta_path.is_file():
        return []

    samples: List[SplitSample] = []
    with meta_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            patient = str(row.get("Patient", "")).strip()
            if not patient:
                continue
            image_path = cfg.wsi_dir() / f"{patient}{cfg.slide_ext}"
            if not image_path.is_file():
                image_path = find_wsi_path(patient, cfg)
            if image_path is None:
                continue
            samples.append(
                SplitSample(
                    sample_id=patient,
                    case_id=patient,
                    split="all",
                    label=str(row.get("Response", row.get("Responder", ""))).strip(),
                    image_path=image_path,
                    extra={k: str(v) for k, v in row.items()},
                )
            )
    return samples


def _discover_hooknet_samples(cfg: DatasetConfig) -> List[SplitSample]:
    if cfg.mask_dir is None or not cfg.mask_dir.is_dir():
        return []

    wsi_map = build_wsi_path_map(cfg)
    samples: List[SplitSample] = []
    for mask_path in sorted(cfg.mask_dir.glob(f"*{cfg.mask_suffix}")):
        slide_id = mask_path.name[: -len(cfg.mask_suffix)]
        image_path = wsi_map.get(slide_id) or find_wsi_path(slide_id, cfg)
        samples.append(
            SplitSample(
                sample_id=slide_id,
                case_id=slide_id,
                split="all",
                label="",
                image_path=image_path,
                mask_path=mask_path.resolve(),
                extra={"wsi_path": str(image_path) if image_path else ""},
            )
        )
    return samples


def _discover_slide_samples(cfg: DatasetConfig) -> List[SplitSample]:
    if cfg.discover_from_masks:
        return _discover_hooknet_samples(cfg)

    samples: List[SplitSample] = []
    if cfg.is_patch_hdf:
        search_dir = cfg.hdf_dir()
        pattern = cfg.glob
        for path in sorted(search_dir.glob(pattern)):
            sid = path.stem
            samples.append(
                SplitSample(
                    sample_id=sid,
                    case_id=sid,
                    split="all",
                    label="",
                    image_path=path.resolve(),
                    extra={"hdf_path": str(path)},
                )
            )
        return samples

    if cfg.recursive_wsi or cfg.wsi_dirs:
        for slide_id, path in sorted(build_wsi_path_map(cfg).items()):
            samples.append(
                SplitSample(
                    sample_id=slide_id,
                    case_id=slide_id,
                    split="all",
                    label="",
                    image_path=path,
                    extra={"wsi_path": str(path)},
                )
            )
        return samples

    search_dir = cfg.wsi_dir()
    for path in sorted(search_dir.glob(f"*{cfg.slide_ext}")):
        sid = _slide_id_from_wsi_name(path)
        samples.append(
            SplitSample(
                sample_id=sid,
                case_id=sid,
                split="all",
                label="",
                image_path=path.resolve(),
                extra={"wsi_path": str(path)},
            )
        )
    return samples


def load_split_samples(
    cfg: DatasetConfig,
    split_filter: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[SplitSample]:
    """Load samples from split CSV; fall back to directory discovery for slide-level datasets."""
    if cfg.manifest_format == "immunocto_compact":
        samples = _load_immunocto_compact_manifest(
            cfg,
            split_filter=split_filter,
            max_samples=max_samples,
        )
    else:
        samples = _load_split_csv(cfg, split_filter=split_filter, max_samples=max_samples)
    if not samples and cfg.metadata_path is not None:
        samples = _load_metadata_samples(cfg)
    if not samples:
        samples = _discover_slide_samples(cfg)
    if split_filter and split_filter.lower() not in {"all", "*", ""} and not (
        samples and all(s.split.lower() == "all" for s in samples)
    ):
        want = _normalize_split_name(split_filter)
        samples = [s for s in samples if s.split.lower() == want]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
