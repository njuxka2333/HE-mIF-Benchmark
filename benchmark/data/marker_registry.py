"""Marker panels, channel order, and canonical-name matching for virtual staining.

Task 1 of the benchmark scores generated mIF against real mIF **per marker**, so
every comparison depends on knowing ``output_channel_index -> marker_name`` for
each generator, and on matching those names to the evaluation cohort's panel.

Channel ORDER is the training/inference emission order
------------------------------------------------------
Index ``i`` is taken from each model's own training stack or official eval
script, not from a paper table of protein names. ``verified=True`` means that
source list has been transcribed. A Cell-niches argmax diagonal is a transfer
check, not the definition of the order. A wrong order silently scores the
wrong markers. :func:`resolve_channel_map` refuses unverified panels unless
``allow_unverified=True``.

Geometry
--------
``MODEL_GEOMETRY`` is the Cell-niches Task-1 **generator** HE feed (256 µm
window): paper-native virtual-stain inference, not the encoder / bag tile.
``feed=mosaic`` chops the resampled window into ``input_size`` tiles with no
overlap; ``feed=whole`` sends the full resampled window to the wrapper (ROSIE
then 8 px stitch; GigaTIME then 256 px windows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "MODEL_GEOMETRY",
    "MODEL_PANELS",
    "DATASET_PANELS",
    "ModelPanel",
    "DatasetPanel",
    "canonical_marker",
    "resolve_channel_map",
    "ChannelMatch",
]


# --------------------------------------------------------------------------
# Canonical marker naming
# --------------------------------------------------------------------------
# Deliberate equivalence decisions, not string matching. Each group maps every
# spelling we have seen onto one canonical token.
_ALIASES: dict[str, str] = {}


def _alias(canonical: str, *spellings: str) -> None:
    _ALIASES[_norm(canonical)] = canonical
    for s in spellings:
        _ALIASES[_norm(s)] = canonical


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# Nuclear/DNA stains are treated as one canonical channel: different dyes, same
# target (nuclear DNA). This is the loosest equivalence here — flag it in the paper.
_alias("DNA", "DAPI", "Hoechst", "Hoechst1", "DRAQ5", "nuclear")
_alias("CD3", "CD3e", "CD3ε")
_alias("CD8", "CD8a", "CD8A")
_alias("CD4")
_alias("CD20")
_alias("CD45")
_alias("CD45RO")
_alias("CD56")
_alias("CD68")
_alias("CD163")
_alias("CD31")
_alias("CD16")
_alias("CD11c")
_alias("CD14")
_alias("CD21")
_alias("CD34")
_alias("CD38")
_alias("CD39")
_alias("CD40")
_alias("CD44")
_alias("CD66")
_alias("CD79a")
_alias("CD138")
_alias("CD141")
_alias("FOXP3", "FoxP3", "Foxp3")
_alias("Ki67", "Ki-67", "MKI67")
_alias("PD-1", "PD1")
_alias("PD-L1", "PDL1", "PD_L1")
_alias("SMA", "aSMA", "alpha-SMA", "αSMA", "ACTA2")
_alias("E-cadherin", "ECad", "Ecadherin", "E-Cad", "CDH1")
_alias("PanCK", "Pan-CK", "panCK", "CK", "Cytokeratin", "pan-cytokeratin")
_alias("GranzymeB", "Granzyme B", "GZMB", "GranzymeB")
_alias("HLA-DR", "HLADR")
_alias("HLA-ABC", "HLAABC")
_alias("HLA-A", "HLAA")
_alias("HLA-E", "HLAE")
_alias("Vimentin", "VIM")
_alias("Podoplanin", "PDPN")
_alias("EpCAM")
_alias("Keratin8_18", "Keratin8/18", "CK8/18")
_alias("CollagenIV", "Collagen IV", "COL4")
_alias("Tryptase")
_alias("Caspase3", "Caspase 3", "CASP3", "cleaved caspase 3", "Caspase3-D")
_alias("PHH3", "pHH3", "phospho-histone H3", "PHH3-B")
_alias("T-bet", "Tbet", "TBX21")
_alias("Transgelin", "TAGLN")
_alias("Actin", "ACTB", "Actin-D")
_alias("MPO")
_alias("PCNA")
_alias("ATM")
_alias("TP63", "p63")
_alias("IFNg", "IFN-g", "IFNγ")
_alias("IDO1")
_alias("LAG3")
_alias("ICOS")
_alias("GATA3")
_alias("Gal3", "Galectin-3", "LGALS3")
_alias("BCL2", "BCL-2")
_alias("TIGIT")
_alias("VISTA")
_alias("gp100", "PMEL")
_alias("MelanA", "Melan-A", "MLANA")
_alias("S100", "S100B")
_alias("SOX10")
# Non-marker channels that must never be scored.
_alias("__background__", "TRITC", "Cy5", "AF", "autofluorescence")


def canonical_marker(name: str) -> str:
    """Canonical token for a marker spelling; unknown names pass through as given."""
    return _ALIASES.get(_norm(name), str(name))


def is_background_channel(name: str) -> bool:
    return canonical_marker(name) == "__background__"


# --------------------------------------------------------------------------
# Per-model output panels and geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPanel:
    """Ordered output channels of one HE->mIF generator."""

    model: str
    channels: tuple[str, ...]
    verified: bool
    source: str
    note: str = ""

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def scored_indices(self) -> list[int]:
        """Channel indices that name a real marker (background channels dropped)."""
        return [i for i, c in enumerate(self.channels) if not is_background_channel(c)]


# Cell-niches Task-1 generator geometry (256 µm window).
# ROSIE: 40x / 0.25 mpp, whole FOV so the wrapper does paper 8 px stitch
#   (128 context → 8×-downsampled map). Mosaic of 128 px tiles would hit the
#   classifier-broadcast path and paint 16 constant blocks.
# GigaTIME: paper 256 px @ 0.5 mpp. Whole 512 canvas; wrapper tiles at 256.
# MIPHEI / HEMIT / HistoPlexer: training operating point (unchanged).
MODEL_GEOMETRY: dict[str, dict[str, float | int | str]] = {
    "rosie": {"mpp": 0.25, "input_size": 128, "feed": "whole"},
    "mipheivit": {"mpp": 0.5, "input_size": 256, "feed": "mosaic"},
    "gigatime": {"mpp": 0.5, "input_size": 256, "feed": "whole"},
    "hemit": {"mpp": 0.25, "input_size": 1024, "feed": "whole"},
    "histoplexer": {"mpp": 0.25, "input_size": 1024, "feed": "whole"},
}


MODEL_PANELS: dict[str, ModelPanel] = {
    "hemit": ModelPanel(
        model="hemit",
        channels=("DAPI", "CD3", "panCK"),
        verified=True,
        source="HEMIT Pix2pix_DualBranch/post_process.py tif_composite ([:,:,0] DAPI, [:,:,1] CD3, [:,:,2] panCK)",
        note=(
            "Checkpoint emits 3 channels. Cell-niches OOD: panCK is the only "
            "structured head; CD3 vs GT CD3 remains weak (~0.05 Pearson)."
        ),
    ),
    "histoplexer": ModelPanel(
        model="histoplexer",
        channels=(
            "CD16", "CD20", "CD3", "CD31", "CD8a", "gp100",
            "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10",
        ),
        verified=True,
        source="HistoPlexer demo/model/config.json markers; downstream index_select([1,2,8])=CD20,CD3,MelanA",
        note=(
            "Training IMC stack order, not a documentation sort. Cell-niches OOD "
            "has no usable unique-stain signal (max |r|~0.10)."
        ),
    ),
    "mipheivit": ModelPanel(
        model="mipheivit",
        channels=(
            "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
            "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "Ki67", "Pan-CK", "SMA",
        ),
        verified=True,
        source="MIPHEI-ViT configs/data/orion.yaml targ_channel_names (training emission order)",
        note=(
            "Last three channels are Ki67, Pan-CK, SMA. Confirmed on Cell-niches: "
            "GT CK/CD3/CD68 argmax onto Pan-CK/CD3e/CD68 (48-patch mean Pearson). "
            "CD4/CD8 may argmax onto CD45 out-of-domain; that is lineage correlation, not a swapped index."
        ),
    ),
    "gigatime": ModelPanel(
        model="gigatime",
        channels=(
            "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
            "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
            "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B",
            "Transgelin",
        ),
        verified=True,
        source="GigaTIME/scripts/db_train.py common_channel_list (COMET file order; same list in gigatime_testing.ipynb)",
        note=(
            "TRITC/Cy5 are background at indices 1–2, not last. Earlier Table-1-then-"
            "background-last order was wrong and has been replaced."
        ),
    ),
    "rosie": ModelPanel(
        model="rosie",
        channels=(
            "DAPI", "CD45", "CD68", "CD14", "PD1", "FoxP3", "CD8", "HLA-DR", "PanCK",
            "CD3e", "CD4", "aSMA", "CD31", "Vimentin", "CD45RO", "Ki67", "CD20",
            "CD11c", "Podoplanin", "PDL1", "GranzymeB", "CD38", "CD141", "CD21",
            "CD163", "BCL2", "LAG3", "EpCAM", "CD44", "ICOS", "GATA3", "Gal3", "CD39",
            "CD34", "TIGIT", "ECad", "CD40", "VISTA", "HLA-A", "MPO", "PCNA", "ATM",
            "TP63", "IFNg", "Keratin8/18", "IDO1", "CD79a", "HLA-E", "CollagenIV",
            "CD66",
        ),
        verified=True,
        source="ROSIE README / HuggingFace model card 50-channel list (train.py all_biomarkers head order)",
        note=(
            "process_exp.py ALL_BIOMARKER_LABELS is a different, longer list — not "
            "this checkpoint. Cell-niches OOD: CD68 hits CD68; PanCK→epithelial "
            "cluster; CD3→leukocyte cluster."
        ),
    ),
}


# --------------------------------------------------------------------------
# Evaluation cohort panels
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetPanel:
    dataset: str
    channels: tuple[str, ...]
    verified: bool
    source: str
    note: str = ""


DATASET_PANELS: dict[str, DatasetPanel] = {
    "orion_crc": DatasetPanel(
        dataset="orion_crc",
        channels=(
            "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
            "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "Ki67", "Pan-CK", "SMA",
        ),
        verified=True,
        source="MIPHEI-ViT configs/data/orion.yaml (same order as the released generator)",
        note="MIPHEI-ViT trained on this cohort - evaluate held-out slides only.",
    ),
    "cell_niches": DatasetPanel(
        dataset="cell_niches",
        channels=(
            "CD3", "CD4", "CD8", "CD20", "CD56", "CD68", "CD163", "FOXP3",
            "Granzyme B", "PD-1", "PD-L1", "CK",
        ),
        verified=True,
        source="Nat Commun 2025 s41467-025-65783-z (NSCLC 12-plex); TIFF order written by extract.py",
        note=(
            "Patch TIFFs are stacked in this order (see datasets/cell_niches_paired_patches/"
            "channel_order.json). No autofluorescence channel; no nuclear channel in the panel."
        ),
    ),
}


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelMatch:
    canonical: str
    model_index: int
    dataset_index: int
    model_name: str
    dataset_name: str


def resolve_channel_map(
    model: str,
    dataset: str,
    *,
    allow_unverified: bool = False,
    model_panels: Mapping[str, ModelPanel] | None = None,
    dataset_panels: Mapping[str, DatasetPanel] | None = None,
) -> list[ChannelMatch]:
    """Markers a model and cohort share, as index pairs.

    Raises unless both panels are marked verified, or ``allow_unverified=True``.
    Background channels are never matched.
    """
    mp = (model_panels or MODEL_PANELS)[model]
    dp = (dataset_panels or DATASET_PANELS)[dataset]

    if not allow_unverified and not (mp.verified and dp.verified):
        unverified = [n for n, ok in ((model, mp.verified), (dataset, dp.verified)) if not ok]
        raise ValueError(
            f"channel order unverified for {unverified}; a wrong order yields "
            "plausible metrics for the wrong markers. Verify against the emitted "
            "tensors and set verified=True, or pass allow_unverified=True for a "
            "clearly-labelled provisional run."
        )

    ds_index: dict[str, int] = {}
    for j, name in enumerate(dp.channels):
        ds_index.setdefault(canonical_marker(name), j)

    matches: list[ChannelMatch] = []
    seen: set[str] = set()
    for i, name in enumerate(mp.channels):
        canon = canonical_marker(name)
        if canon == "__background__" or canon in seen:
            continue
        j = ds_index.get(canon)
        if j is None:
            continue
        seen.add(canon)
        matches.append(
            ChannelMatch(
                canonical=canon,
                model_index=i,
                dataset_index=j,
                model_name=name,
                dataset_name=dp.channels[j],
            )
        )
    return matches


def coverage_table(
    dataset: str,
    models: Sequence[str] | None = None,
    *,
    allow_unverified: bool = True,
) -> dict[str, int]:
    """How many of a cohort's markers each model can be scored on."""
    models = list(models or MODEL_PANELS)
    return {
        m: len(resolve_channel_map(m, dataset, allow_unverified=allow_unverified))
        for m in models
    }
