"""
Output path helpers for feature extraction runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from models.registry import get_output_layout


def format_output_paths(
    layout_name: str,
    feat_dir: str,
    model: str,
    slide_id: str,
    config_path: str | None = None,
) -> Dict[str, str]:
    """Resolve concrete file paths for a slide under the chosen layout."""
    layout = get_output_layout(layout_name, config_path)
    paths_cfg = layout.get("paths") or {}
    ctx = {"feat_dir": feat_dir, "model": model, "slide_id": slide_id}
    resolved: Dict[str, str] = {}
    for key, pattern in paths_cfg.items():
        resolved[key] = pattern.format(**ctx)
    return resolved


def ensure_output_dirs(layout_name: str, feat_dir: str, model: str, config_path: str | None = None) -> None:
    """Create parent directories for all paths in a layout (excluding manifest files)."""
    dummy = format_output_paths(layout_name, feat_dir, model, "_placeholder", config_path)
    for key, path in dummy.items():
        if key == "manifest":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)


def legacy_prepath_dirs(feat_dir: str, model: str) -> Dict[str, Path]:
    """Original PrePATH directory structure used by feature_extraction/run_fp_fast.py."""
    base = Path(feat_dir)
    return {
        "pt_dir": base / "pt_files" / model,
        "h5_dir": base / "h5_files" / model,
    }


def describe_layouts() -> str:
    lines = [
        "Current default (prepath_legacy):",
        "  {feat_dir}/pt_files/{model}/{slide_id}.pt   # feature tensors",
        "  {feat_dir}/h5_files/{model}/{slide_id}.h5  # coords metadata",
        "",
        "Proposed (benchmark_v1):",
        "  {feat_dir}/{model}/embeddings/{slide_id}.pt  # vector features (linear/ABMIL)",
        "  {feat_dir}/{model}/spatial/{slide_id}.pt     # dense maps (UNETR / mIF)",
        "  {feat_dir}/{model}/metadata/{slide_id}.h5    # coords + run metadata",
        "  {feat_dir}/{model}/manifest.json               # run-level index",
    ]
    return "\n".join(lines)
