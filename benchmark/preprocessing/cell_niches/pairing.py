"""Load Cell-niches HE–mIF FOV pairs, Aligned polygons, and marker files."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import shapely.wkb
from shapely.geometry.base import BaseGeometry

from data.marker_registry import DATASET_PANELS, canonical_marker

from .io import index_wsis

PANEL_CHANNELS: tuple[str, ...] = DATASET_PANELS["cell_niches"].channels
MIF_MPP_UM = 0.325
HE_MPP_FALLBACK_UM = 0.11


def deserialize_wkb(df: pd.DataFrame) -> pd.DataFrame:
    load_wkb = lambda x: shapely.wkb.load(io.BytesIO(x))
    return df.assign(geom=df.geom.apply(load_wkb))


def stain_to_panel_name(stain: str) -> str | None:
    canon = canonical_marker(stain)
    for name in PANEL_CHANNELS:
        if canonical_marker(name) == canon:
            return name
    return None


@dataclass
class FOVPair:
    fov_id: str
    entity: str
    he_path: Path
    marker_paths: dict[str, Path]
    dapi_path: Path | None
    dapi_stain: str | None
    aligned: BaseGeometry
    regions_path: Path
    child_rows: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def missing_markers(self) -> list[str]:
        return [c for c in PANEL_CHANNELS if c not in self.marker_paths]


def load_aligned_polygon(regions_root: Path, fov_id: str) -> tuple[BaseGeometry, Path]:
    folder = Path(regions_root) / f"wsi_uuid={fov_id}"
    parquets = sorted(folder.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no region parquet under {folder}")
    path = parquets[0]
    df = pd.read_parquet(path).pipe(deserialize_wkb)
    hit = df.loc[df["region"].astype(str).str.fullmatch("Aligned", case=False)]
    if hit.empty:
        raise KeyError(f"no Aligned region in {path}")
    return hit.iloc[0].geom, path


def load_fov_pairs(
    root: Path,
    *,
    wsi_index: dict[str, Path] | None = None,
) -> list[FOVPair]:
    root = Path(root)
    meta = pd.read_csv(root / "tier1/metadata/he_to_mif.csv")
    wsi_index = wsi_index or index_wsis(root / "tier1/wsis")
    regions_root = root / "tier2/regions"
    fovs: list[FOVPair] = []
    for fov_id, rows in meta.groupby("parent", sort=True):
        entity = str(rows["ENTITY"].iloc[0])
        he_path = wsi_index.get(str(fov_id))
        if he_path is None:
            raise FileNotFoundError(f"HE TIFF missing for parent {fov_id}")
        marker_paths: dict[str, Path] = {}
        dapi_candidates: dict[str, Path] = {}
        for rec in rows.itertuples(index=False):
            stain = str(rec.child_stain)
            child_path = wsi_index.get(str(rec.child))
            if child_path is None:
                continue
            panel_name = stain_to_panel_name(stain)
            if panel_name is not None:
                marker_paths[panel_name] = child_path
            if stain.upper().startswith("DAPI"):
                dapi_candidates[stain] = child_path
        dapi_path = None
        dapi_stain = None
        for key in ("DAPI1", "DAPI2", "DAPI3"):
            if key in dapi_candidates:
                dapi_path = dapi_candidates[key]
                dapi_stain = key
                break
        aligned, regions_path = load_aligned_polygon(regions_root, str(fov_id))
        fovs.append(
            FOVPair(
                fov_id=str(fov_id),
                entity=entity,
                he_path=he_path,
                marker_paths=marker_paths,
                dapi_path=dapi_path,
                dapi_stain=dapi_stain,
                aligned=aligned,
                regions_path=regions_path,
                child_rows=rows.reset_index(drop=True),
            )
        )
    return fovs
