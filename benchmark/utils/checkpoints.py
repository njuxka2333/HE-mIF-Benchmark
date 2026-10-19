"""Resolve model checkpoint paths from benchmark_models.yaml."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

from models.schema import HubCheckpointConfig, ModelConfig
from utils.hf_auth import hf_token
from utils.paths import PREPATH_ROOT, REPO_ROOT, checkpoint_root, checkpoint_search_dirs

DEFAULT_CKPT_DIR = checkpoint_root()
DOWNLOAD_CACHE_DIR = DEFAULT_CKPT_DIR / "downloads"

_LEGACY_CKPT_NAMES = {
    "rosie": "rosie.pth",
    "hemmit": "hemmit_net_G.pth",
    "hemit": "hemmit_net_G.pth",
    "gigatime": "gigatime.pth",
    "mipheivit": "mipheivit.ckpt",
    "miphei": "mipheivit.ckpt",
    "histoplexer": "histoplexer.pth",
}


def resolve_path(raw: str | Path, base: Path = PREPATH_ROOT) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = []
    for ckpt_dir in checkpoint_search_dirs():
        candidates.append((ckpt_dir / path).resolve())
        candidates.append((ckpt_dir / path.name).resolve())
    candidates.extend(
        [
            (base / path).resolve(),
            (REPO_ROOT / path).resolve(),
            (REPO_ROOT / path.name).resolve(),
        ]
    )
    name = path.name
    if name in {"histoplexer.pt", "pytorch_model.pt", "histoplexer.pth"}:
        candidates.append(REPO_ROOT / "HistoPlexer" / "demo" / "model" / "pytorch_model.pt")
    if name in {"rosie_best_model_single.pth", "best_model_single.pth", "rosie.pth"}:
        candidates.append(REPO_ROOT / "rosie" / "best_model_single.pth")
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return candidates[0]


def _cache_path(hub: HubCheckpointConfig) -> Path:
    name = hub.filename or Path(hub.url or "weights").name
    return DOWNLOAD_CACHE_DIR / name


def download_url_checkpoint(hub: HubCheckpointConfig) -> str:
    if not hub.url:
        raise ValueError("hub.url is required for direct URL downloads.")
    dest = _cache_path(hub)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        urllib.request.urlretrieve(hub.url, dest)
    return str(dest.resolve())


def _hf_hub_local_path(hub: HubCheckpointConfig) -> Optional[Path]:
    """Return a cached Hugging Face weight file without network access."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    token = hf_token()
    revision = hub.revision or "main"
    cache_dirs: list[Optional[str]] = []
    if hub.cache_dir:
        cache_dirs.append(hub.cache_dir)
    cache_dirs.append(str(DOWNLOAD_CACHE_DIR / "hf"))
    cache_dirs.append(None)  # default ~/.cache/huggingface

    seen: set[str] = set()
    for cache_dir in cache_dirs:
        key = cache_dir or "__default__"
        if key in seen:
            continue
        seen.add(key)
        kwargs = {
            "repo_id": hub.repo_id,
            "filename": hub.filename,
            "revision": revision,
            "token": token,
            "local_files_only": True,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        try:
            path = hf_hub_download(**kwargs)
        except Exception:
            continue
        resolved = Path(path).resolve()
        if resolved.is_file():
            return resolved
    return None


def download_hub_checkpoint(hub: HubCheckpointConfig) -> str:
    if hub.url and not hub.repo_id:
        return download_url_checkpoint(hub)
    if not hub.repo_id or not hub.filename:
        raise ValueError(
            "hub.repo_id and hub.filename are required for Hugging Face Hub downloads."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hub checkpoint loading requires huggingface_hub. "
            "Install with: pip install huggingface_hub"
        ) from exc

    dest = _cache_path(hub)
    if dest.is_file():
        return str(dest.resolve())

    cached = _hf_hub_local_path(hub)
    if cached is not None:
        return str(cached)

    kwargs = {
        "repo_id": hub.repo_id,
        "filename": hub.filename,
        "revision": hub.revision or "main",
        "token": hf_token(),
    }
    if hub.cache_dir:
        kwargs["cache_dir"] = hub.cache_dir
    path = hf_hub_download(**kwargs)
    return str(Path(path).resolve())


def hub_reference(hub: HubCheckpointConfig) -> str:
    if hub.url and not hub.repo_id:
        return hub.url
    rev = hub.revision or "main"
    return f"hf://{hub.repo_id}@{rev}/{hub.filename}"


def resolve_model_checkpoint(
    cfg: ModelConfig,
    *,
    must_exist: bool = True,
    config_path: Optional[str] = None,
) -> str:
    """
    Resolve checkpoint path for a model.

    Priority:
      1. existing explicit ``checkpoint_path`` under ``checkpoints/``
      2. ``hub`` (Hugging Face ``repo_id`` or a direct artifact ``url``)
      3. legacy filename from ``checkpoint_key`` under ``checkpoints/``
    """
    del config_path

    configured_path: Path | None = None
    if cfg.checkpoint_path:
        configured_path = resolve_path(cfg.checkpoint_path)
        if configured_path.is_file():
            return str(configured_path.resolve())

    if cfg.uses_hub():
        hub = cfg.hub
        assert hub is not None
        if must_exist:
            return download_hub_checkpoint(hub)
        return hub_reference(hub)

    if configured_path is not None:
        if must_exist:
            raise FileNotFoundError(
                f"Checkpoint not found for model '{cfg.name}': {configured_path}\n"
                "Set checkpoint_path or hub in configs/benchmark_models.yaml, "
                "or place the file in checkpoints/ (see README)."
            )
        return str(configured_path)

    key = (cfg.checkpoint_key or cfg.name).lower()
    legacy_name = _LEGACY_CKPT_NAMES.get(key)
    if legacy_name:
        path = DEFAULT_CKPT_DIR / legacy_name
        if must_exist and not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found for model '{cfg.name}': {path}\n"
                "Download weights into checkpoints/ or set hub in "
                f"configs/benchmark_models.yaml.\nExpected directory: {DEFAULT_CKPT_DIR}"
            )
        return str(path.resolve())

    raise FileNotFoundError(
        f"No checkpoint configured for model '{cfg.name}'. "
        "Set checkpoint_path or hub in configs/benchmark_models.yaml."
    )
