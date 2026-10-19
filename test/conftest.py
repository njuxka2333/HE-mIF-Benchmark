"""Pytest fixtures for PrePATH integration tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
PREPATH_ROOT = BENCHMARK_ROOT
OUTPUT_ROOT = TESTS_DIR / "outputs"
EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"

MODELS = ["rosie", "hemit", "gigatime", "mipheivit", "uni2", "histoplexer"]
TEST_DATA_CONFIG = REPO_ROOT / "configs" / "datasets_test.yaml"
TEST_DATASETS = ["immunocto", "pathocell", "hooknet_tls", "tcga_coad"]
PATCH_DATASETS = ["immunocto"]
WSI_DATASETS = ["hooknet_tls", "tcga_coad"]
SAMPLE_PREVIEW_LIMIT = 20

WSI_SMOKE_CONFIG = "abmil_tcga_coad_rosie_40x_s128.yaml"

if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    """Fail fast when pytest is not using the project conda env."""
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise pytest.UsageError(
            "torch/torchvision failed to import in this interpreter:\n"
            f"  python: {sys.executable}\n"
            f"  error: {exc}\n"
            "Activate the project env and run:\n"
            "  conda activate hemif\n"
            "  python -m pytest test/ -v\n"
            "Do not use bare `pytest` if it points at another Python."
        ) from exc


@pytest.fixture(scope="session")
def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session", autouse=True)
def _mock_aslide():
    if "Aslide" not in sys.modules:
        sys.modules["Aslide"] = mock.MagicMock()


@pytest.fixture
def sync_save(monkeypatch):
    """Use synchronous saves so subprocess timing does not flake in tests."""
    import torch

    from utils.file_utils import save_hdf5, save_patch_bag_h5

    def _save_feature(path, feature):
        torch.save(feature, path)

    def _save_h5(output_path, asset_dict, attr_dict=None):
        save_hdf5(output_path, asset_dict=asset_dict, attr_dict=attr_dict, mode="w")

    def _save_patch_bag(output_path, feature_obj, coords, masks=None, attr_dict=None, **kwargs):
        save_patch_bag_h5(
            output_path,
            feature_obj,
            coords,
            masks,
            attr_dict=attr_dict,
            feature_dtype=kwargs.get("feature_dtype", "float16"),
            mask_dtype=kwargs.get("mask_dtype", "uint8"),
        )

    monkeypatch.setattr("feature_extraction.extract_features_fp.save_feature_subprocess", _save_feature)
    monkeypatch.setattr("feature_extraction.extract_features_fp.save_hdf5_subprocess", _save_h5)
    monkeypatch.setattr("feature_extraction.extract_features_fp.save_patch_bag_h5_subprocess", _save_patch_bag)


def rosie_weights_path() -> Path:
    from utils.paths import ROSIE_CHECKPOINT

    return ROSIE_CHECKPOINT


def hemit_weights_path() -> Path:
    from utils.paths import HEMIT_CHECKPOINT

    return HEMIT_CHECKPOINT


def local_checkpoint_path(model: str) -> Path | None:
    """Return a local checkpoint file for ``model``, or None if not on disk."""
    from models.registry import get_model_config
    from utils.checkpoints import DEFAULT_CKPT_DIR, _cache_path, resolve_path
    from utils.paths import HEMIT_CHECKPOINT, ROSIE_CHECKPOINT

    canonical = {
        "rosie": ROSIE_CHECKPOINT,
        "hemit": HEMIT_CHECKPOINT,
    }
    if model in canonical and canonical[model].is_file():
        return canonical[model]

    cfg = get_model_config(model)
    if cfg.checkpoint_path:
        path = resolve_path(cfg.checkpoint_path)
        if path.is_file():
            return path

    if cfg.uses_hub():
        hub = cfg.hub
        assert hub is not None
        cached = _cache_path(hub)
        if cached.is_file():
            return cached
        legacy = DEFAULT_CKPT_DIR / (hub.filename or "")
        if legacy.is_file():
            return legacy
        if hub.repo_id and hub.filename:
            from utils.checkpoints import _hf_hub_local_path

            cached_hf = _hf_hub_local_path(hub)
            if cached_hf is not None:
                return cached_hf
    return None


def ensure_model_weights(model: str) -> None:
    """Skip tests when checkpoint files are not available locally."""
    if local_checkpoint_path(model) is not None:
        return
    if model in {"gigatime", "mipheivit"}:
        pytest.skip(
            f"Model weights not cached locally for '{model}'. "
            "Download via Hugging Face (run `huggingface-cli login` for GigaTIME) "
            "or place the file in checkpoints/ (or $HE_MIF_CKPT_DIR)."
        )
    pytest.skip(
        f"Model weights not found for '{model}'. "
        f"ROSIE: {rosie_weights_path()} | HEMIT: {hemit_weights_path()}"
    )


def expected_output_paths(feat_dir: Path, model: str, sample_id: str) -> dict[str, Path]:
    return {
        "pt": feat_dir / "pt_files" / model / f"{sample_id}.pt",
        "h5": feat_dir / "h5_files" / model / f"{sample_id}.h5",
        "mask": feat_dir / "mask_files" / model / f"{sample_id}.pt",
    }


def assert_output_layout(
    feat_dir: Path,
    model: str,
    sample_id: str,
    *,
    save_masks: bool,
    feature_storage: str = "legacy_pt",
) -> None:
    paths = expected_output_paths(feat_dir, model, sample_id)
    if feature_storage == "patch_h5":
        assert paths["h5"].is_file(), paths["h5"]
        assert not paths["pt"].exists(), paths["pt"]
        return
    assert paths["pt"].is_file(), paths["pt"]
    assert paths["h5"].is_file(), paths["h5"]
    if save_masks:
        assert paths["mask"].is_file(), paths["mask"]


def assert_output_contents(
    feat_dir: Path,
    model: str,
    sample_id: str,
    *,
    save_masks: bool,
    expect_multi_layer: bool = False,
    feature_storage: str = "legacy_pt",
) -> None:
    """Validate saved feature tensors and coord bags, not just file existence."""
    import h5py
    import torch

    from utils.file_utils import is_patch_bag_h5

    paths = expected_output_paths(feat_dir, model, sample_id)
    if feature_storage == "patch_h5":
        assert is_patch_bag_h5(paths["h5"])
        with h5py.File(paths["h5"], "r") as handle:
            assert "coords" in handle
            assert len(handle["coords"]) >= 1
            assert "features" in handle
            if expect_multi_layer:
                assert isinstance(handle["features"], h5py.Group)
                assert handle["features"].keys()
            if save_masks:
                assert "masks" in handle
                assert len(handle["masks"]) >= 1
        return

    feature_obj = torch.load(paths["pt"], map_location="cpu")
    if expect_multi_layer:
        assert isinstance(feature_obj, dict), feature_obj
        assert feature_obj, "empty feature dict"
        for tensor in feature_obj.values():
            assert tensor.ndim >= 2, tensor.shape
    else:
        assert hasattr(feature_obj, "shape") or isinstance(feature_obj, dict)

    with h5py.File(paths["h5"], "r") as handle:
        assert "coords" in handle
        assert len(handle["coords"]) >= 1

    if save_masks:
        masks = torch.load(paths["mask"], map_location="cpu")
        assert masks is not None


def apply_test_dataset_config(exp):
    """Point an experiment at portable fixtures under test/test_data/."""
    from data.loader import get_dataset_config

    exp.dataset_config = get_dataset_config(exp.dataset, str(TEST_DATA_CONFIG))
    return exp


class MockWsiHandle:
    """Minimal WSI stand-in for Whole_Slide_Bag_FP.read_region."""

    def __init__(self, patch_size: int = 256, fill: int = 200):
        self.patch_size = patch_size
        self.fill = fill
        self.properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}

    def read_region(self, coord, level, size):
        from PIL import Image

        w, h = int(size[0]), int(size[1])
        return Image.new("RGB", (w, h), color=(self.fill, self.fill, self.fill))


def patch_test_dataset_loader(monkeypatch):
    """Route dataset resolution and sample loading through test fixtures."""
    import data.loader as dataset_loader
    import utils.experiment_config as experiment_loader
    from data.loader import get_dataset_config as _get_dataset_config
    from data.loader import load_split_samples as _load_split_samples

    def _get(name: str, config_path=None):
        return _get_dataset_config(name, str(TEST_DATA_CONFIG))

    def _load(cfg, split_filter=None, max_samples=None):
        if max_samples is None:
            max_samples = 1
        return _load_split_samples(cfg, split_filter=split_filter, max_samples=max_samples)

    monkeypatch.setattr(experiment_loader, "get_dataset_config", _get)
    monkeypatch.setattr(dataset_loader, "get_dataset_config", _get)
    monkeypatch.setattr("feature_extraction.extract_features_fp.load_split_samples", _load)


def load_dataset_samples(
    dataset: str,
    *,
    split_filter: str | None = None,
    max_samples: int = SAMPLE_PREVIEW_LIMIT,
    config_path: Path | None = None,
):
    """Load dataset samples from portable test fixtures."""
    import pytest

    from data.loader import get_dataset_config, load_split_samples

    cfg_path = str(config_path or TEST_DATA_CONFIG)
    cfg = get_dataset_config(dataset, cfg_path)
    samples = load_split_samples(cfg, split_filter=split_filter, max_samples=max_samples)

    if not samples:
        pytest.skip(f"No samples resolved for {dataset} under {cfg.dataset_dir}")
    return cfg, samples


def resolved_samples(samples, limit: int = SAMPLE_PREVIEW_LIMIT):
    """Return up to ``limit`` samples whose image paths exist on disk."""
    return [s for s in samples[:limit] if s.image_path and s.image_path.is_file()]
