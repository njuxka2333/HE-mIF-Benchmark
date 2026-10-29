"""HookNet-TLS XML annotation loading and semantic mask extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.loader import DatasetConfig


HOOKNET_TLS_LABELS = {
    "background": 0,
    "tls": 1,
    "gc": 2,
}

OFFICIAL_HOOKNET_TLS_LABELS = {
    "background": 0,
    "tls": 1,
    "gc": 2,
    "rest": 3,
    "other": 3,
    "inflammation": 3,
    "tumor": 3,
    "lymph node": 3,
}

def xml_path_for_slide(cfg: DatasetConfig, slide_id: str) -> Path | None:
    if not cfg.annotations_subdir:
        return None
    path = cfg.dataset_dir / cfg.annotations_subdir / f"{slide_id}.xml"
    return path if path.is_file() else None

def load_wsa(xml_path: Path, *, backend: str = "openslide"):
    """Load ASAP XML annotations for tls/gc; return None if file has no tls/gc regions."""
    from wholeslidedata.annotation.parsers import PARSERS
    from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation

    parser = PARSERS["asap"](labels=HOOKNET_TLS_LABELS)
    if not parser.parse(str(xml_path)):
        return None
    return WholeSlideAnnotation(
        str(xml_path),
        labels=HOOKNET_TLS_LABELS,
        parser="asap",
    )

def load_official_wsa(xml_path: Path, *, backend: str = "openslide"):
    """Load ASAP XML with official tls/gc/rest label mapping."""
    from wholeslidedata.annotation.parsers import PARSERS
    from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation

    parser = PARSERS["asap"](labels=OFFICIAL_HOOKNET_TLS_LABELS)
    if not parser.parse(str(xml_path)):
        return None
    return WholeSlideAnnotation(
        str(xml_path),
        labels=OFFICIAL_HOOKNET_TLS_LABELS,
        parser="asap",
    )

def semantic_mask_from_xml(
    wsa,
    wsi,
    x: int,
    y: int,
    patch_size: int,
    spacing: float,
    *,
    center: bool = False,
) -> np.ndarray:
    from wholeslidedata.samplers.patchlabelsampler import SegmentationPatchLabelSampler

    ratio = int(wsi.get_downsampling_from_spacing(spacing))
    if center:
        point = (int(x), int(y))
    else:
        half = (patch_size * ratio) // 2
        point = (int(x) + half, int(y) + half)

    sampler = SegmentationPatchLabelSampler()
    return sampler.sample(wsa, point, (patch_size, patch_size), ratio).astype(np.int64)
