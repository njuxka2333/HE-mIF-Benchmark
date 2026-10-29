"""Protocol layout helpers for Pre_Features extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from feature_extraction.fold_placement import cleanup_unique_wsi_files
from feature_extraction.protocol import (
    dataset_protocol,
    encoder_dir,
    layer_root,
    parse_layer_tokens,
    protocol_layer_feat_dirs,
    run_id,
    run_root_from_feat_dir,
)
from utils.experiment_config import EXPERIMENTS_DIR, _default_layer_feat_dirs
from utils.paths import PRE_FEATURES_ROOT


def test_apply_protocol_layers_expands_encoder_and_output():
    from feature_extraction.extract import apply_protocol_layers
    from utils.experiment_config import load_experiment_config

    exp = load_experiment_config(EXPERIMENTS_DIR / "abmil_tcga_coad_rosie_40x_s128.yaml")
    roots = apply_protocol_layers(exp, ["stage3", "mif0"])
    run = PRE_FEATURES_ROOT / "abmil_tcga_coad_rosie_40x_s128"
    assert exp.feat_dir == run / "encoder_stage3"
    assert exp.layer_feat_dirs["mif0"] == run / "output_layers"
    assert roots == [run / "encoder_stage3", run / "output_layers"]
    assert exp.split_layer_feat_dirs is True


def test_run_ids_and_layer_roots(tmp_path):
    assert run_id("tcga_coad", "rosie") == "abmil_tcga_coad_rosie_40x_s128"
    assert run_id("immunocto", "uni2") == "linear_probe_immunocto_uni2_20x_s32"
    ds = dataset_protocol("tcga_coad")
    assert ds.kind == "wsi_cv"
    assert ds.n_folds == 5
    run_root = tmp_path / "abmil_tcga_coad_rosie_40x_s128"
    assert layer_root(run_root, "rosie", "stage3") == run_root / "encoder_stage3"
    assert layer_root(run_root, "rosie", "mif0") == run_root / "output_layers"
    assert encoder_dir(run_root, "stage3") == run_root / "encoder_stage3"


def test_parse_layer_tokens_all_and_yaml():
    assert parse_layer_tokens("all", "rosie") == ["stage3", "mif0"]
    assert parse_layer_tokens("encoder", "uni2") == ["z4"]
    assert parse_layer_tokens("yaml", "gigatime", yaml_layers=["x4_0"]) == ["x4_0"]
    assert parse_layer_tokens("output", "hemit") == ["mif"]
    with pytest.raises(ValueError):
        parse_layer_tokens("output", "uni2")


def test_run_root_from_feat_dir(tmp_path):
    run_root = tmp_path / "abmil_tcga_coad_rosie_40x_s128"
    assert run_root_from_feat_dir(run_root / "encoder_stage3") == run_root
    assert run_root_from_feat_dir(run_root / "output_layers") == run_root
    assert run_root_from_feat_dir(run_root) == run_root


def test_protocol_layer_feat_dirs_maps_encoder_and_output(tmp_path):
    feat_dir = tmp_path / "abmil_tcga_coad_rosie_40x_s128" / "encoder_stage3"
    mapping = protocol_layer_feat_dirs(feat_dir, "rosie", ["stage3", "mif0"])
    assert mapping["stage3"] == feat_dir
    assert mapping["mif0"] == feat_dir.parent / "output_layers"


def test_experiment_config_default_layer_dirs_use_protocol(tmp_path):
    feat_dir = tmp_path / "run" / "encoder_z4"
    mapping = _default_layer_feat_dirs(feat_dir, ["z4", "mif"], model="mipheivit")
    assert mapping["z4"] == feat_dir
    assert mapping["mif"] == feat_dir.parent / "output_layers"


def test_place_wsi_split_writes_train_test_and_drops_folds(tmp_path):
    from feature_extraction.fold_placement import (
        empty_stats,
        index_wsi_bags,
        place_wsi_split,
        remove_fold_layout,
    )

    layer = tmp_path / "encoder_z4"
    model = "uni2"
    train_src = layer / "fold_0" / "train" / "h5_files" / model / "slideA.h5"
    train_src.parent.mkdir(parents=True)
    train_src.write_bytes(b"train-bag")
    test_src = layer / "h5_files" / model / "slideB.h5"
    test_src.parent.mkdir(parents=True)
    test_src.write_bytes(b"test-bag")
    split_csv = tmp_path / "extract.csv"
    pd.DataFrame(
        [
            {"slide_id": "slideA", "split": "train"},
            {"slide_id": "slideB", "split": "test"},
        ]
    ).to_csv(split_csv, index=False)

    stats = empty_stats()
    place_wsi_split(
        index_wsi_bags(layer, model),
        layer,
        model,
        split_csv,
        stats,
    )
    train_h5 = layer / "train" / "h5_files" / model / "slideA.h5"
    test_h5 = layer / "test" / "h5_files" / model / "slideB.h5"
    assert train_h5.is_file() and test_h5.is_file()
    assert train_h5.stat().st_ino == train_src.stat().st_ino
    assert test_h5.stat().st_ino == test_src.stat().st_ino
    dropped = remove_fold_layout(layer)
    assert "fold_0" in dropped
    assert train_h5.is_file() and test_h5.is_file()
    assert not (layer / "fold_0").exists()
    assert {p.name for p in layer.iterdir()} == {"train", "test"}


def test_place_cv_wsi_hardlinks_valid_equals_test(tmp_path):
    layer = tmp_path / "encoder_z4"
    model = "uni2"
    src = layer / "h5_files" / model / "slideA.h5"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"bag")
    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()
    for fold in range(5):
        split = "test" if fold == 0 else "train"
        pd.DataFrame(
            [{"slide_id": "slideA", "split": split, "case_id": "c"}]
        ).to_csv(fold_dir / f"fold{fold}.csv", index=False)

    ds = dataset_protocol("tcga_coad")
    # Use local fold CSVs instead of production ones.
    from feature_extraction.fold_placement import place_cv_wsi, index_wsi_bags, empty_stats

    stats = empty_stats()
    place_cv_wsi(
        index_wsi_bags(layer, model),
        layer,
        model,
        [fold_dir / f"fold{i}.csv" for i in range(5)],
        stats,
        valid_equals_test=True,
    )
    train_h5 = layer / "fold_1" / "train" / "h5_files" / model / "slideA.h5"
    test_h5 = layer / "fold_0" / "test" / "h5_files" / model / "slideA.h5"
    valid_h5 = layer / "fold_0" / "valid" / "h5_files" / model / "slideA.h5"
    assert train_h5.is_file()
    assert test_h5.is_file()
    assert valid_h5.is_file()
    assert test_h5.stat().st_ino == valid_h5.stat().st_ino
    assert src.stat().st_ino == train_h5.stat().st_ino
    removed = cleanup_unique_wsi_files(layer, model)
    assert removed == 1
    assert not src.exists()
    assert train_h5.is_file()
    assert ds.n_folds == 5
