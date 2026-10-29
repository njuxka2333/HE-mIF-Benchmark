"""Config loading: datasets registry and experiment YAMLs."""

from __future__ import annotations

from conftest import EXPERIMENTS_DIR, MODELS
from utils.paths import CONFIG_DIR, PRE_FEATURES_ROOT
from data.loader import load_dataset_registry
from utils.experiment_config import list_experiment_configs, load_experiment_config
from training.config import load_eval_config
from data.feature_bags import DATASET_REGISTRY
from heads import HEAD_REGISTRY

REPORTED_DATASETS = {
    "pathocell",
    "pathocell_online",
    "lizard",
    "lizard_online",
    "hooknet_tls",
    "tcga_luad",
    "tcga_coad",
    "nucls",
    "immunocto",
    "cell_niches",
}

DOWNSTREAM_DIR = CONFIG_DIR / "downstream"
SCORING_HEADS = {"virtual_stain"}


def test_eval_config_coerces_scientific_notation_optimizer_strings(tmp_path):
    path = tmp_path / "optimizer_numeric.yaml"
    path.write_text(
        """
evaluation:
  task: prognosis
  dataset: classification_features
  head: classification_attention_mil
training:
  optimizer:
    name: Adam
    params:
      lr: 1e-04
      weight_decay: 1e-05
""",
        encoding="utf-8",
    )
    cfg = load_eval_config(path)
    params = cfg.training["optimizer"]["params"]
    assert params["lr"] == 1e-4
    assert params["weight_decay"] == 1e-5
    assert isinstance(params["lr"], float)
    assert isinstance(params["weight_decay"], float)


def test_datasets_registry_has_reported_entries():
    assert set(load_dataset_registry()) == REPORTED_DATASETS


def test_all_experiment_yamls_parse():
    paths = list_experiment_configs()
    assert len(paths) == len(list(EXPERIMENTS_DIR.glob("*.yaml")))
    for path in paths:
        exp = load_experiment_config(path)
        assert exp.model in MODELS, path.name
        assert exp.dataset in load_dataset_registry(), path.name
        assert exp.target_features, path.name
        assert exp.feat_dir.is_absolute()


def test_experiment_feat_dirs_are_under_pre_features():
    for path in list_experiment_configs():
        exp = load_experiment_config(path)
        feat = str(exp.feat_dir)
        assert feat.startswith(str(PRE_FEATURES_ROOT)), feat


def test_all_downstream_yamls_parse():
    paths = sorted(DOWNSTREAM_DIR.rglob("*.yaml"))
    assert paths
    known_datasets = set(DATASET_REGISTRY) | set(load_dataset_registry())
    for path in paths:
        cfg = load_eval_config(path)
        assert cfg.dataset in known_datasets, path.name
        assert cfg.head in HEAD_REGISTRY or cfg.head in SCORING_HEADS, path.name
        feature_config = cfg.data.get("feature_experiment_config")
        if feature_config:
            assert load_experiment_config(feature_config), path.name


def test_every_reported_run_has_a_downstream_yaml():
    models = ("gigatime", "hemit", "histoplexer", "mipheivit", "rosie", "uni2")
    vs_models = models[:-1]  # UNI2 has no virtual-staining layer
    expected = []
    for task in (
        "single_cell_classification/immunocto",
        "single_cell_classification/nucls",
        "prognosis/tcga_coad",
        "prognosis/tcga_luad",
        "tls_region_segmentation/hooknet_tls",
        "cell_semantic_segmentation/pathocell",
        "cell_semantic_segmentation/lizard",
    ):
        for layer in ("encoder", "output"):
            for model in models if layer == "encoder" else vs_models:
                expected.append(DOWNSTREAM_DIR / task / layer / f"{model}.yaml")
    for model in vs_models:
        expected.append(
            DOWNSTREAM_DIR / "virtual_staining" / "cell_niches" / "output" / f"{model}.yaml"
        )
    missing = [str(path) for path in expected if not path.is_file()]
    extra = sorted(
        p for p in set(DOWNSTREAM_DIR.rglob("*.yaml")) - set(expected)
        if p.relative_to(DOWNSTREAM_DIR).parts[0] not in {
            "single_cell_classification_hoptimus",
            "single_cell_classification_pca",
        }
    )
    assert not missing, missing
    assert not extra, extra
    assert len(expected) == 82


def test_linear_probe_selects_on_valid_and_tests_once():
    import yaml

    for path in sorted((DOWNSTREAM_DIR / "single_cell_classification").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        training = raw.get("training") or {}
        assert training.get("evaluate_test") is True, path
        assert training.get("evaluate_test_each_epoch") is not True, path
        select = str(training.get("checkpoint_select_metric") or f"valid_{training.get('primary_metric')}")
        assert select.startswith("valid_"), path
        assert not select.startswith("test_"), path


def test_tls_and_cellseg_validate_on_full_grid():
    import yaml

    for task in ("tls_region_segmentation", "cell_semantic_segmentation"):
        for path in sorted((DOWNSTREAM_DIR / task).rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            training = raw.get("training") or {}
            data = raw.get("data") or {}
            assert str(data.get("valid_split") or "valid") == "valid", path
            assert str(data.get("test_split") or "test") == "test", path
            protocol = str(training.get("online_valid_protocol") or "").strip().lower()
            assert protocol == "full_grid", path
            assert training.get("skip_test") is False, path


def test_vstain_yaml_is_cell_niches_shared():
    import yaml

    from data.marker_registry import MODEL_GEOMETRY
    from metrics.vstain import PATCH_UM, PROTOCOL_ID

    for model, geom in MODEL_GEOMETRY.items():
        path = DOWNSTREAM_DIR / "virtual_staining" / "cell_niches" / "output" / f"{model}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data = raw.get("data") or {}
        exp = raw.get("experiment") or {}
        training = raw.get("training") or {}
        run = raw.get("run") or {}
        assert data.get("protocol") == PROTOCOL_ID, path
        assert float(data.get("patch_um")) == PATCH_UM, path
        assert data.get("feed") == geom["feed"], path
        assert float(exp.get("target_mpp")) == float(geom["mpp"]), path
        assert int(exp.get("target_patch_size")) == int(geom["input_size"]), path
        assert training.get("significance") is True, path
        out = str(run.get("output_dir") or "")
        assert "Evaluation/virtual_staining" in out, path
        assert "/results/" not in out.replace("\\", "/"), path


def test_lab_prefixes_remap_onto_this_clone():
    from utils.paths import DEFAULT_REPO_PREFIX, REPO_ROOT, evaluation_root, relocate_lab_path

    raw = DEFAULT_REPO_PREFIX / "Pre_Features" / "foo"
    assert relocate_lab_path(raw) == REPO_ROOT / "Pre_Features" / "foo"
    assert evaluation_root() == REPO_ROOT / "Evaluation"


def test_cell_niches_raw_root_uses_env(monkeypatch, tmp_path):
    from utils.paths import cell_niches_raw_root

    monkeypatch.setenv("HE_MIF_CELL_NICHES_RAW", str(tmp_path))
    assert cell_niches_raw_root() == tmp_path.resolve()
