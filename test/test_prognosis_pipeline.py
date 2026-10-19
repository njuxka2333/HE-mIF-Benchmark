import numpy as np
import pandas as pd
import torch


def test_mil_feature_loader_reads_patch_bag_h5(tmp_path):
    import h5py

    from data.feature_bags import _load_mil_feature_file, _select_mil_feature

    h5_path = tmp_path / "bag.h5"
    with h5py.File(h5_path, "w") as handle:
        features = handle.create_group("features")
        features.create_dataset("features", data=np.ones((3, 16), dtype=np.float16))
        handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int32))

    feature_obj = _load_mil_feature_file(h5_path)
    tensor = _select_mil_feature(feature_obj, "z4")

    assert tensor.shape == (3, 16)


def test_survival_attention_mil_forward_and_loss():
    from heads import SurvivalAttentionMIL
    from losses import NLLSurvLoss

    model = SurvivalAttentionMIL(input_dim=16, hidden_dim=8, embed_dim=4, n_classes=4)
    hazards, survival, prediction = model(torch.randn(5, 16))
    assert hazards.shape == (1, 4)
    assert survival.shape == (1, 4)
    assert prediction.shape == (1, 1)

    loss = NLLSurvLoss(alpha=0.0)(
        hazards,
        survival,
        torch.tensor([1]),
        torch.tensor([0.0]),
    )
    assert torch.isfinite(loss)


def test_shazam_survival_abmil_forward_bias_free_attention_and_alpha_loss():
    from heads import ShazamSurvivalABMIL
    from losses import NLLSurvLoss

    model = ShazamSurvivalABMIL(input_dim=16, hidden_dim=8, dropout=0.25, n_classes=4)
    assert model.attention[-1].bias is None
    hazards, survival, prediction = model(torch.randn(7, 16))
    assert hazards.shape == (1, 4)
    assert survival.shape == (1, 4)
    assert prediction.shape == (1, 1)
    assert model.last_attention is not None
    loss = NLLSurvLoss(alpha=0.4)(
        hazards,
        survival,
        torch.tensor([2]),
        torch.tensor([0.0]),
    )
    assert torch.isfinite(loss)


def test_shazam_bins_use_split_local_quantiles_with_zero_inf_edges():
    from data.feature_bags import _survival_labels

    split = pd.DataFrame(
        {
            "case_id": [f"p{i}" for i in range(8)],
            "time": [10, 20, 30, 40, 50, 60, 70, 80],
            "event": [1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    labelled, bins = _survival_labels(
        split,
        time_column="time",
        event_column="event",
        n_bins=4,
        event_positive_values=[1],
        bin_frame=split.assign(event=split["event"]),
        bin_policy="shazam_split",
    )
    assert bins[0] == 0.0
    assert np.isinf(bins[-1])
    assert labelled["disc_label"].min() >= 0
    assert labelled["disc_label"].max() <= 3


def test_survival_slide_prediction_unit_emits_one_row_per_slide(tmp_path):
    import h5py
    import yaml

    from data.feature_bags import SurvivalFeatureDataset

    feature_root = tmp_path / "features"
    rows = [
        {"case_id": "case_0", "slide_id": "slide_0a", "split": "train", "time": 10, "event": 1},
        {"case_id": "case_0", "slide_id": "slide_0b", "split": "train", "time": 10, "event": 1},
        {"case_id": "case_1", "slide_id": "slide_1", "split": "train", "time": 20, "event": 1},
        {"case_id": "case_2", "slide_id": "slide_2", "split": "train", "time": 30, "event": 1},
        {"case_id": "case_3", "slide_id": "slide_3", "split": "train", "time": 40, "event": 0},
        {"case_id": "case_4", "slide_id": "slide_4", "split": "train", "time": 50, "event": 1},
        {"case_id": "case_5", "slide_id": "slide_5", "split": "train", "time": 60, "event": 1},
        {"case_id": "case_6", "slide_id": "slide_6", "split": "train", "time": 70, "event": 1},
        {"case_id": "case_7", "slide_id": "slide_7", "split": "train", "time": 80, "event": 1},
    ]
    for row in rows:
        path = feature_root / "train" / "h5_files" / "gigatime" / f"{row['slide_id']}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            features = handle.create_group("features")
            features.create_dataset("x4_0", data=np.ones((2, 4), dtype=np.float16))

    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    feature_cfg = tmp_path / "features.yaml"
    feature_cfg.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "test",
                    "head": "abmil",
                    "model": "gigatime",
                    "dataset": "tcga_coad",
                    "target_mpp": 0.5,
                    "target_patch_size": 256,
                    "target_features": ["x4_0"],
                },
                "run": {"feat_dir": str(feature_root), "feature_storage": "patch_h5"},
            }
        )
    )
    dataset = SurvivalFeatureDataset(
        feature_cfg,
        "train",
        split_csv=split_csv,
        time_column="time",
        event_column="event",
        n_bins=4,
        prediction_unit="slide",
        bin_policy="shazam_split",
        feature_layer="x4_0",
    )
    first = dataset[0]
    cached = dataset[0]
    assert torch.allclose(cached["features"].float(), first["features"].float())
    assert 0 in getattr(dataset, "_bag_cache", {})
    assert any(item["case_id"] == "case_0" for item in dataset)



def test_concordance_index_orders_higher_risk_earlier_events():
    from metrics import concordance_index

    c_index = concordance_index(
        event_times=np.array([1.0, 2.0, 3.0]),
        risks=np.array([3.0, 2.0, 1.0]),
        censorships=np.array([0.0, 0.0, 0.0]),
    )
    assert c_index == 1.0


def test_survival_bootstrap_ci_and_external_risk_threshold():
    from metrics import concordance_index_bootstrap_ci, logrank_by_median_risk

    event_times = np.array([1.0, 2.0, 3.0, 4.0])
    risks = np.array([0.9, 0.7, 0.2, 0.1])
    censorships = np.array([0.0, 0.0, 1.0, 0.0])

    ci = concordance_index_bootstrap_ci(
        event_times,
        risks,
        censorships,
        n_bootstraps=20,
        seed=42,
    )
    assert 0.0 <= ci["c_index_ci_low"] <= ci["c_index_ci_high"] <= 1.0
    assert ci["c_index_bootstrap_n"] > 0

    out = logrank_by_median_risk(
        event_times,
        risks,
        censorships,
        risk_threshold=0.5,
    )
    assert out["risk_threshold"] == 0.5
    assert out["risk_group_high_n"] == 2.0


def test_survival_bins_use_training_outer_limits_and_can_be_reused():
    from data.feature_bags import _survival_labels

    train = pd.DataFrame(
        {"case_id": [f"t{i}" for i in range(8)], "time": [2, 4, 6, 8, 10, 12, 14, 16], "event": [1] * 8}
    )
    cohort = pd.concat(
        [train, pd.DataFrame({"case_id": ["valid"], "time": [1000], "event": [0]})],
        ignore_index=True,
    )
    labelled, bins = _survival_labels(
        cohort,
        time_column="time",
        event_column="event",
        n_bins=4,
        event_positive_values=[1],
        bin_frame=train,
    )
    assert bins[0] == train["time"].min() - 1e-6
    assert bins[-1] == train["time"].max() + 1e-6

    _, reused = _survival_labels(
        train,
        time_column="time",
        event_column="event",
        n_bins=4,
        event_positive_values=[1],
        bin_frame=train,
        bin_boundaries=bins,
    )
    np.testing.assert_array_equal(reused, bins)
    assert labelled.loc[labelled.case_id == "valid", "disc_label"].item() == 3


def test_survival_training_outcome_permutation_keeps_patient_features_fixed(tmp_path):
    import h5py
    import yaml

    from data.feature_bags import SurvivalFeatureDataset

    feature_root = tmp_path / "features"
    rows = []
    for index in range(8):
        case_id = f"case_{index}"
        slide_id = f"slide_{index}"
        rows.append({"case_id": case_id, "slide_id": slide_id, "split": "train", "time": index + 1, "event": index % 2})
        path = feature_root / "train" / "h5_files" / "gigatime" / f"{slide_id}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            features = handle.create_group("features")
            features.create_dataset("x4_0", data=np.full((2, 4), index, dtype=np.float16))

    split_csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    feature_cfg = tmp_path / "features.yaml"
    feature_cfg.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "test",
                    "head": "abmil",
                    "model": "gigatime",
                    "dataset": "tcga_coad",
                    "target_mpp": 0.5,
                    "target_patch_size": 256,
                    "target_features": ["x4_0"],
                },
                "run": {"feat_dir": str(feature_root), "feature_storage": "patch_h5"},
            }
        )
    )
    dataset = SurvivalFeatureDataset(
        feature_cfg,
        "train",
        split_csv=split_csv,
        time_column="time",
        event_column="event",
        n_bins=4,
        feature_layer="x4_0",
        permute_training_outcomes=True,
        outcome_permutation_seed=42,
    )
    assert dataset.outcome_permutation["enabled"] is True
    assert dataset.outcome_permutation["n_changed"] > 0
    for item in dataset:
        expected = int(item["case_id"].split("_")[-1])
        assert torch.all(item["features"] == expected)

    try:
        SurvivalFeatureDataset(
            feature_cfg,
            "valid",
            split_csv=split_csv,
            time_column="time",
            event_column="event",
            n_bins=4,
            permute_training_outcomes=True,
        )
    except ValueError as error:
        assert "only for the training split" in str(error)
    else:
        raise AssertionError("Expected non-training outcome permutation to fail")


def test_survival_delayed_checkpointing_does_not_count_prestart_epochs(monkeypatch, tmp_path):
    """The first eligible checkpoint cannot be epoch 0 or trigger immediate stopping."""
    from training import offline_train
    from training.config import EvalConfig

    class Dataset:
        num_classes = 4
        feature_normalize = "none"

        def __len__(self):
            return 1

    model = torch.nn.Linear(1, 1)
    cfg = EvalConfig(
        id="delayed",
        task="prognosis_survival",
        dataset="survival_features",
        head="survival_attention_mil",
        output_dir=tmp_path,
        data={},
        training={
            "max_epochs": 9,
            "checkpoint_selection_start_epoch": 4,
            "early_stopping_start_epoch": 8,
            "early_stopping_patience": 4,
            "primary_metric": "c_index",
            "primary_metric_mode": "max",
            "gradient_accumulation": 1,
            "weighted_sample": False,
            "seed": 42,
            "device": "cpu",
        },
    )
    monkeypatch.setattr(offline_train, "_prepare_output_dirs", lambda _: {"logs": tmp_path, "validation": tmp_path, "test": tmp_path})
    monkeypatch.setattr(offline_train, "_prepare_survival_datasets", lambda _: (Dataset(), Dataset(), {"splits": {}}))
    monkeypatch.setattr(offline_train, "_write_dataset_audit", lambda *args: None)
    monkeypatch.setattr(offline_train, "_make_loader", lambda *args, **kwargs: [None])
    monkeypatch.setattr(offline_train, "_make_head", lambda *args: model)
    monkeypatch.setattr(offline_train, "_loss_fn", lambda *args: object())
    monkeypatch.setattr(offline_train, "_survival_optimizer", lambda *args: torch.optim.SGD(model.parameters(), lr=0.1))
    monkeypatch.setattr(offline_train, "_make_survival_scheduler", lambda *args: None)
    monkeypatch.setattr(offline_train, "train_one_survival_epoch", lambda *args: {"loss": 1.0, "c_index": 0.5})
    monkeypatch.setattr(offline_train, "evaluate_model", lambda *args: {"loss": 1.0, "c_index": 0.5})
    monkeypatch.setattr(offline_train, "_write_metrics", lambda *args: None)
    monkeypatch.setattr(offline_train, "_log", lambda *args: None)

    saved_epochs = []
    monkeypatch.setattr(
        torch,
        "save",
        lambda payload, path: saved_epochs.append(payload["epoch"]),
    )
    offline_train.run_survival(cfg)
    assert saved_epochs == [4]


def test_classification_attention_mil_forward_and_metrics():
    from heads import ClassificationAttentionMIL
    from metrics import classification_metrics

    model = ClassificationAttentionMIL(input_dim=16, hidden_dim=8, embed_dim=4, n_classes=2)
    logits, probabilities, prediction = model(torch.randn(5, 16))
    assert logits.shape == (1, 2)
    assert probabilities.shape == (1, 2)
    assert prediction.shape == (1, 1)

    out = classification_metrics(
        labels=np.array([0, 1, 1]),
        predictions=np.array([0, 1, 0]),
        probabilities=np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]]),
        num_classes=2,
    )
    assert out["accuracy"] > 0
    assert 0 <= out["auc"] <= 1
