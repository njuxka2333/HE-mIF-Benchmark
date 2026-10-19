"""Canonical Pre_Features layout: layers, geometry, run ids, split paths.

Source of truth is ``configs/extraction/protocol.yaml``. Extraction writes:

  {run_id}/
    encoder_{layer}/
      train|valid|test/                                      # linear-probe / wsi_split
      fold_{0-4}/{train,valid,test}/h5_files/{model}/{slide}.h5  # WSI CV
    output_layers/   # mif / mif0; UNI2 has none
      ... same ...
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import yaml

from utils.paths import CONFIG_DIR, resolve_project_path

PROTOCOL_YAML = CONFIG_DIR / "extraction" / "protocol.yaml"
_PROTOCOL_OVERRIDE: Path | None = None

LayerKind = Literal["encoder", "output"]
DatasetKind = Literal["linear_probe", "wsi_cv", "wsi_split", "online"]


def set_protocol_yaml(path: Path | None) -> None:
    global _PROTOCOL_OVERRIDE
    _PROTOCOL_OVERRIDE = Path(path) if path is not None else None
    load_protocol.cache_clear()


@dataclass(frozen=True)
class ModelProtocol:
    name: str
    encoder: str
    output: str | None
    lp_mpp: float
    lp_patch: int
    lp_tag: str
    wsi_mpp: float
    wsi_patch: int
    wsi_tag: str


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    kind: DatasetKind
    splits: tuple[str, ...]
    n_folds: int
    valid_equals_test: bool
    fold_csv_template: str | None
    split_csv: Path | None
    online: bool

    def fold_csvs(self) -> list[Path]:
        if not self.fold_csv_template:
            return []
        return [Path(self.fold_csv_template.format(fold=i)) for i in range(self.n_folds)]


@dataclass(frozen=True)
class ExtractionProtocol:
    models: dict[str, ModelProtocol]
    datasets: dict[str, DatasetProtocol]


def _require(mapping: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{ctx} missing {key!r}")
    return mapping[key]


def _resolve_split_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = resolve_project_path(path)
    return path


def _resolve_split_template(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(resolve_project_path(path))


@lru_cache(maxsize=1)
def load_protocol(path: Path | None = None) -> ExtractionProtocol:
    yaml_path = path or _PROTOCOL_OVERRIDE or PROTOCOL_YAML
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    models: dict[str, ModelProtocol] = {}
    for name, spec in (raw.get("models") or {}).items():
        lp = spec.get("linear_probe") or {}
        wsi = spec.get("wsi") or {}
        output = spec.get("output")
        models[name] = ModelProtocol(
            name=name,
            encoder=str(_require(spec, "encoder", f"models.{name}")),
            output=None if output in {None, "", "null"} else str(output),
            lp_mpp=float(lp["target_mpp"]),
            lp_patch=int(lp["target_patch_size"]),
            lp_tag=str(lp["tag"]),
            wsi_mpp=float(wsi["target_mpp"]),
            wsi_patch=int(wsi["target_patch_size"]),
            wsi_tag=str(wsi["tag"]),
        )
    datasets: dict[str, DatasetProtocol] = {}
    for name, spec in (raw.get("datasets") or {}).items():
        kind = str(spec.get("kind") or "online")
        if kind not in {"linear_probe", "wsi_cv", "wsi_split", "online"}:
            raise ValueError(
                f"datasets.{name}.kind must be linear_probe|wsi_cv|wsi_split|online, got {kind!r}"
            )
        datasets[name] = DatasetProtocol(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            splits=tuple(str(x) for x in (spec.get("splits") or ())),
            n_folds=int(spec.get("n_folds") or 0),
            valid_equals_test=bool(spec.get("valid_equals_test", False)),
            fold_csv_template=_resolve_split_template(spec.get("fold_csv")),
            split_csv=_resolve_split_path(spec.get("split_csv")),
            online=bool(spec.get("online", kind == "online")),
        )
    return ExtractionProtocol(models=models, datasets=datasets)


def model_protocol(model: str, protocol: ExtractionProtocol | None = None) -> ModelProtocol:
    proto = protocol or load_protocol()
    try:
        return proto.models[model]
    except KeyError as exc:
        raise KeyError(f"Unknown model {model!r}; known={sorted(proto.models)}") from exc


def dataset_protocol(dataset: str, protocol: ExtractionProtocol | None = None) -> DatasetProtocol:
    proto = protocol or load_protocol()
    try:
        return proto.datasets[dataset]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {dataset!r}; known={sorted(proto.datasets)}") from exc


def is_output_layer(model: str, layer: str, protocol: ExtractionProtocol | None = None) -> bool:
    spec = model_protocol(model, protocol)
    return spec.output is not None and layer == spec.output


def run_id(dataset: str, model: str, protocol: ExtractionProtocol | None = None) -> str:
    proto = protocol or load_protocol()
    ds = dataset_protocol(dataset, proto)
    spec = model_protocol(model, proto)
    if ds.kind == "linear_probe":
        return f"linear_probe_{dataset}_{model}_{spec.lp_tag}"
    return f"abmil_{dataset}_{model}_{spec.wsi_tag}"


def run_root_from_feat_dir(feat_dir: Path) -> Path:
    """Infer ``Pre_Features/{run_id}`` from a training or extraction feat_dir."""
    name = feat_dir.name
    if name == "output_layers" or name.startswith("encoder_"):
        return feat_dir.parent
    return feat_dir


def encoder_dir(run_root: Path, layer: str) -> Path:
    return run_root / f"encoder_{layer}"


def output_dir(run_root: Path) -> Path:
    return run_root / "output_layers"


def layer_root(run_root: Path, model: str, layer: str, protocol: ExtractionProtocol | None = None) -> Path:
    if is_output_layer(model, layer, protocol):
        return output_dir(run_root)
    return encoder_dir(run_root, layer)


def protocol_layer_feat_dirs(
    feat_dir: Path,
    model: str,
    layers: Sequence[str],
    protocol: ExtractionProtocol | None = None,
) -> dict[str, Path]:
    """Map layer names to encoder_* / output_layers roots."""
    run_root = run_root_from_feat_dir(feat_dir)
    return {layer: layer_root(run_root, model, layer, protocol) for layer in layers}


def parse_layer_tokens(
    raw: str | Sequence[str] | None,
    model: str,
    *,
    yaml_layers: Sequence[str] | None = None,
    protocol: ExtractionProtocol | None = None,
) -> list[str]:
    """Resolve CLI/config layer tokens to concrete layer names.

    Tokens: ``all``, ``encoder``, ``output``, ``yaml``, or a layer name (``z4``, ``mif``).
    """
    spec = model_protocol(model, protocol)
    if raw is None or raw == "" or raw == "yaml":
        layers = [str(x) for x in (yaml_layers or [spec.encoder])]
        return _dedupe(layers)

    tokens: list[str]
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    else:
        tokens = [str(part).strip() for part in raw if str(part).strip()]

    resolved: list[str] = []
    for token in tokens:
        key = token.lower()
        if key in {"all", "both"}:
            resolved.append(spec.encoder)
            if spec.output:
                resolved.append(spec.output)
        elif key in {"encoder", "enc"}:
            resolved.append(spec.encoder)
        elif key in {"output", "mif", "output_layers"}:
            if not spec.output:
                raise ValueError(f"model {model} has no output/mIF layer")
            resolved.append(spec.output)
        elif key == "yaml":
            resolved.extend(str(x) for x in (yaml_layers or [spec.encoder]))
        else:
            resolved.append(token)
    return _dedupe(resolved)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def wsi_slide_h5(layer_root_path: Path, model: str, slide_id: str, *, fold: int, split: str) -> Path:
    return layer_root_path / f"fold_{fold}" / split / "h5_files" / model / f"{slide_id}.h5"


def unique_wsi_h5(layer_root_path: Path, model: str, slide_id: str) -> Path:
    return layer_root_path / "h5_files" / model / f"{slide_id}.h5"


def find_existing_wsi_h5(layer_root_path: Path, model: str, slide_id: str) -> Path | None:
    unique = unique_wsi_h5(layer_root_path, model, slide_id)
    if unique.is_file():
        return unique
    for split in ("train", "valid", "test"):
        path = layer_root_path / split / "h5_files" / model / f"{slide_id}.h5"
        if path.is_file():
            return path
    for path in sorted(layer_root_path.glob(f"fold_*/**/h5_files/{model}/{slide_id}.h5")):
        if path.is_file():
            return path
    return None
