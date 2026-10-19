"""Repository path resolution for HE-mIF-Benchmark."""

from __future__ import annotations

import os
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parent
# YAML paths such as ../Evaluation still resolve from the code package.
PREPATH_ROOT = BENCHMARK_ROOT
CONFIG_DIR = REPO_ROOT / "configs"
PRE_FEATURES_ROOT = REPO_ROOT / "Pre_Features"
SPLITS_ROOT = REPO_ROOT / "splits"
# Machine-local image root used in configs/datasets.yaml. Override with HE_MIF_DATASETS_ROOT.
DEFAULT_DATASETS_ROOT = Path("/data3/houjunlin/fxx/datasets")
# Absolute prefix from this lab's clone. Remapped to REPO_ROOT so leftover YAML still works.
DEFAULT_REPO_PREFIX = Path("/data3/houjunlin/fxx/HE-mIF-Benchmark/HE-mIF-Benchmark")
DEFAULT_CELL_NICHES_RAW = Path("/jhcnas6/Cytology/Spatial_Proteomics_Temp/Cell-niches-data")


def resolve_project_path(raw: str | Path, base: Path | None = None) -> Path:
    """Resolve a repo-relative path.

    ``configs/``, ``splits/``, ``results/``, ``Pre_Features/``, and ``checkpoints/``
    are rooted at the GitHub repo. Everything else is relative to ``base``
    (the ``benchmark/`` package, so ``../Evaluation`` still works).
    """
    path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if path.is_absolute():
        return path
    posix = path.as_posix()
    for prefix in ("configs/", "splits/", "results/", "Pre_Features/", "checkpoints/"):
        if posix == prefix.rstrip("/") or posix.startswith(prefix):
            return (REPO_ROOT / path).resolve()
    return ((base or PREPATH_ROOT) / path).resolve()


def datasets_root() -> Path:
    env = os.environ.get("HE_MIF_DATASETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATASETS_ROOT


def relocate_dataset_path(raw: str | Path) -> Path:
    """Resolve a dataset path, then remap the default prefix if HE_MIF_DATASETS_ROOT is set."""
    path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    env_root = os.environ.get("HE_MIF_DATASETS_ROOT")
    if not env_root:
        return path
    try:
        return Path(env_root).expanduser().resolve() / path.relative_to(DEFAULT_DATASETS_ROOT)
    except ValueError:
        return path


def relocate_lab_path(raw: str | Path) -> Path:
    """Remap this lab's absolute prefixes onto the current clone and dataset root."""
    path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    try:
        path = REPO_ROOT / path.relative_to(DEFAULT_REPO_PREFIX)
    except ValueError:
        pass
    return relocate_dataset_path(path)


DATASETS_ROOT = datasets_root()


def evaluation_root() -> Path:
    """Scratch train / figure dir for a local rerun (``<repo>/Evaluation``)."""
    return REPO_ROOT / "Evaluation"


def tls_fixed_grid_root() -> Path:
    return relocate_dataset_path(DEFAULT_DATASETS_ROOT / "HookNet_TLS_fixed_grid_ps1024")


def cell_niches_raw_root() -> Path:
    """Unfiltered Cell-niches HE/mIF source. Set ``HE_MIF_CELL_NICHES_RAW`` off this machine."""
    env = os.environ.get("HE_MIF_CELL_NICHES_RAW")
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_CELL_NICHES_RAW.is_dir():
        return DEFAULT_CELL_NICHES_RAW
    raise FileNotFoundError(
        "Cell-niches raw data not found. Set HE_MIF_CELL_NICHES_RAW to the "
        "Cell-niches-data directory (paired HE/mIF FOVs)."
    )


def checkpoint_root() -> Path:
    """Foundation-model weights for models that are not on Hugging Face Hub.

    Override with ``HE_MIF_CKPT_DIR``. Default is ``<repo>/checkpoints``.
    """
    env = os.environ.get("HE_MIF_CKPT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "checkpoints").resolve()


def checkpoint_search_dirs() -> list[Path]:
    """Directories to search for a local weight file."""
    return [checkpoint_root()]


CKPT_DIR = checkpoint_root()
ROSIE_CHECKPOINT = CKPT_DIR / "rosie_best_model_single.pth"
HEMIT_CHECKPOINT = CKPT_DIR / "hemit_v1.pth"
UNI2_CHECKPOINT = CKPT_DIR / "uni2_pytorch_model.bin"


def reference_model_path(name: str) -> Path:
    """Directory of an upstream model checkout cloned into the repo root."""
    mapping = {
        "rosie": REPO_ROOT / "rosie",
        "hemit": REPO_ROOT / "HEMIT" / "Pix2pix_DualBranch",
        "gigatime": REPO_ROOT / "GigaTIME",
        "mipheivit": REPO_ROOT / "MIPHEI-ViT",
        "histoplexer": REPO_ROOT / "HistoPlexer",
    }
    return mapping[name.lower()]
