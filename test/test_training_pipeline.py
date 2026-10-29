"""Smoke tests for downstream evaluation pipeline components."""

from __future__ import annotations

import numpy as np
import torch
from unittest.mock import patch

from losses import ClassExcludedMeanLoss, EMAInverseClassFrequencyLoss, FocalLoss


def test_pathocell_ema_updates_once_per_loss_call():
    criterion = EMAInverseClassFrequencyLoss(
        torch.nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none"),
        exclude_class=[0],
        num_classes=4,
        alpha=0.99,
        class_weighting=True,
    )
    logits = torch.randn(3, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (3, 8, 8))
    criterion(logits, target).backward()
    assert int(criterion.ema_updates.item()) == 1


def test_pathocell_loss_large_dense_batch_keeps_finite_denominator():
    class UnitPixelLoss(torch.nn.Module):
        def forward(self, pred, target):
            return torch.ones_like(target, dtype=torch.float32)

    criterion = EMAInverseClassFrequencyLoss(
        UnitPixelLoss(),
        exclude_class=[],
        num_classes=2,
        class_weighting=False,
    )
    logits = torch.zeros(2, 2, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.ones(2, 256, 256, dtype=torch.long)
    loss = criterion(logits, target)
    assert torch.isfinite(loss)
    assert float(loss) == 1.0


def test_pathocell_loss_can_disable_distributed_sync_for_rank0_evaluation():
    criterion = EMAInverseClassFrequencyLoss(
        torch.nn.CrossEntropyLoss(reduction="none"),
        exclude_class=[0],
        num_classes=2,
        class_weighting=False,
    )
    criterion.distributed_sync = False
    logits = torch.randn(1, 2, 8, 8)
    target = torch.ones(1, 8, 8, dtype=torch.long)
    with patch("losses.dist.is_initialized", return_value=True), patch(
        "losses.dist.all_reduce", side_effect=AssertionError("unexpected collective")
    ):
        loss = criterion(logits, target)
    assert torch.isfinite(loss)


def test_training_rng_state_roundtrip():
    import random

    from training.offline_train import _restore_rng_state, _rng_state, set_seed

    set_seed(123)
    state = _rng_state()
    observed = (random.random(), np.random.random(), torch.rand(4))
    _restore_rng_state(state)
    expected = (random.random(), np.random.random(), torch.rand(4))

    assert observed[:2] == expected[:2]
    assert torch.equal(observed[2], expected[2])


def test_pure_focal_loss_uses_class_excluded_wrapper_without_ema(tmp_path):
    from training.config import load_eval_config
    from training.offline_train import _loss_fn

    path = tmp_path / "focal.yaml"
    path.write_text(
        """
evaluation:
  id: focal_loss_unit
  task: tls_region_segmentation
  dataset: hooknet_tls
  head: unetr
training:
  loss_fn:
    name: FocalLoss
    params:
      gamma: 2.0
      reduction: none
      ignore_index: 255
    exclude_classes: [0]
    class_weighting: false
    use_ema_wrapper: false
run:
  output_dir: /tmp/focal_loss_unit
""",
        encoding="utf-8",
    )
    cfg = load_eval_config(path)
    criterion = _loss_fn(cfg, num_classes=4)
    assert isinstance(criterion, ClassExcludedMeanLoss)
    assert isinstance(criterion.loss_fn, FocalLoss)
    assert not hasattr(criterion, "ema_frequencies")

    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16))
    loss = criterion(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_feature_unetr_forward_shapes():
    from heads import FeatureUNETR

    layers = ["z1", "z2"]
    features = {
        "z1": torch.randn(2, 16, 32, 32),
        "z2": torch.randn(2, 32, 16, 16),
    }
    model = FeatureUNETR(
        feature_layers=layers,
        in_channels=[16, 32],
        adapter_channels=[8, 16],
        num_classes=4,
    )
    image = torch.randn(2, 3, 64, 64)
    out = model(features, image=image, output_size=(64, 64))
    assert out.shape == (2, 4, 64, 64)


def test_feature_unetr_z0_rgb_with_four_encoder_layers():
    from heads import FeatureUNETR, forward_segmentation_head

    feature_layers = ["z1", "z2", "z3", "z4"]
    features = {
        "z1": torch.randn(2, 64, 16, 16),
        "z2": torch.randn(2, 64, 16, 16),
        "z3": torch.randn(2, 64, 16, 16),
        "z4": torch.randn(2, 64, 16, 16),
    }
    image = torch.randn(2, 3, 224, 224)
    masks = torch.zeros(2, 224, 224, dtype=torch.long)
    model = FeatureUNETR(
        feature_layers=feature_layers,
        in_channels=[64, 64, 64, 64],
        adapter_channels=[128, 256, 512, 512],
        num_classes=5,
    )
    out = model(features, image=image, output_size=(224, 224))
    assert out.shape == (2, 5, 224, 224)
    routed = forward_segmentation_head(model, features, masks, images=image)
    assert routed.shape == (2, 5, 224, 224)


def test_feature_unetr_use_z0_requires_image():
    from heads import FeatureUNETR
    import pytest

    model = FeatureUNETR(
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[32, 32, 32, 32],
        num_classes=3,
    )
    features = {name: torch.randn(1, 32, 8, 8) for name in model.feature_layers}
    with pytest.raises(ValueError, match="requires an RGB image"):
        model(features, output_size=(64, 64))


def test_pixel_metrics_report_macro_micro_and_classwise():
    from metrics import SegmentationMetricAccumulator

    target = np.array([[0, 1, 1], [0, 2, 2]])
    pred = np.array([[0, 1, 2], [0, 2, 0]])
    metrics = SegmentationMetricAccumulator(num_classes=3).update(pred, target)
    assert metrics is None

    acc = SegmentationMetricAccumulator(num_classes=3)
    acc.update(pred, target)
    out = acc.compute()
    assert "f1_macro" in out
    assert "f1_micro" in out
    assert "class_0/iou" not in out
    assert "class_1/iou" in out
    assert "class_2/precision" in out


def test_instance_level_metrics_match_pfmb_protocol():
    from metrics import InstanceLevelMetricAccumulator

    semantic_gt = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    instance_gt = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    pred_probs = np.zeros((3, 4, 4), dtype=np.float32)
    pred_probs[1, semantic_gt == 1] = 0.9
    pred_probs[2, semantic_gt == 2] = 0.9
    pred_probs[0] = 0.05

    label_dict = {0: "background", 1: "class_1", 2: "class_2"}
    acc = InstanceLevelMetricAccumulator(label_dict=label_dict)
    acc.update(pred_probs, semantic_gt, instance_gt)
    out = acc.compute()
    assert out["f1_score_macro"] > 0.99
    assert out["instance_count"] == 2.0
    assert out["class_1/f1_score"] > 0.99
    assert out["class_2/f1_score"] > 0.99


def test_json_string_label_dict_keys_zero_instance_f1_until_coerced():
    """Regression: PathoCell JSON keys are strings; PFMB metrics require ints."""
    from data.loader import coerce_label_dict_keys
    from metrics import InstanceLevelMetricAccumulator, get_instance_level_labels

    semantic_gt = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    instance_gt = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    pred_probs = np.zeros((3, 4, 4), dtype=np.float32)
    pred_probs[1, semantic_gt == 1] = 0.9
    pred_probs[2, semantic_gt == 2] = 0.9
    pred_probs[0] = 0.05

    json_label_dict = {"0": "background", "1": "B cells", "2": "T cells"}
    broken = InstanceLevelMetricAccumulator(label_dict=json_label_dict)
    broken.update(pred_probs, semantic_gt, instance_gt)
    broken_out = broken.compute()
    assert broken_out["f1_score_macro"] == 0.0
    assert "class_1/f1_score" in broken_out
    _, y_pred_broken = get_instance_level_labels(
        pred_probs, semantic_gt, instance_gt, json_label_dict
    )
    assert set(y_pred_broken.tolist()) <= {0}

    coerced = coerce_label_dict_keys(json_label_dict)
    assert coerced == {0: "background", 1: "B cells", 2: "T cells"}
    acc = InstanceLevelMetricAccumulator(label_dict=coerced)
    acc.update(pred_probs, semantic_gt, instance_gt)
    out = acc.compute()
    assert out["f1_score_macro"] > 0.99
    assert out["B cells/f1_score"] > 0.99
    assert out["T cells/f1_score"] > 0.99


def test_cell_auprc_and_pixel_pearson_metrics():
    from metrics import CellAUPRCAccumulator, PixelPearsonAccumulator

    semantic_gt = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    instance_gt = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    pred_probs = np.zeros((3, 4, 4), dtype=np.float32)
    pred_probs[1, semantic_gt == 1] = 0.9
    pred_probs[2, semantic_gt == 2] = 0.9
    pred_probs[0] = 0.05

    label_dict = {0: "background", 1: "class_1", 2: "class_2"}
    auprc = CellAUPRCAccumulator(label_dict=label_dict, num_classes=3)
    auprc.update(pred_probs, semantic_gt, instance_gt)
    auprc_out = auprc.compute()
    assert auprc_out["cell_auprc_macro"] > 0.99
    assert auprc_out["cell_auprc_instance_count"] == 2.0

    pearson = PixelPearsonAccumulator(num_classes=3)
    pearson.update(pred_probs, semantic_gt)
    pearson_out = pearson.compute(label_dict=label_dict)
    assert pearson_out["pixel_pearson_r"] > 0.5
    assert pearson_out["class_1/pearson_r"] > 0.5
    assert pearson_out["class_2/pearson_r"] > 0.5


def test_pfmb_instance_metric_helpers():
    from metrics import (
        aggregate_pfmb_instance_metrics,
        format_pfmb_instance_validation_log,
        uses_pfmb_instance_primary_metric,
    )

    cfg = type(
        "Cfg",
        (),
        {"task": "cell_semantic_segmentation", "training": {"primary_metric": "f1_score_macro"}},
    )()
    assert uses_pfmb_instance_primary_metric(cfg) is True

    tls_cfg = type(
        "Cfg",
        (),
        {"task": "tls_region_segmentation", "training": {"primary_metric": "f1_macro"}},
    )()
    assert uses_pfmb_instance_primary_metric(tls_cfg) is False

    metrics = {
        "f1_score_macro": 0.301,
        "precision_macro": 0.28,
        "recall_macro": 0.25,
        "instance_count": 42.0,
        "f1_macro": 0.15,
        "iou_macro": 0.09,
        "loss": 1.2,
    }
    agg = aggregate_pfmb_instance_metrics(metrics)
    assert "f1_macro" not in agg
    assert agg["f1_score_macro"] == 0.301
    log_line = format_pfmb_instance_validation_log(metrics)
    assert "valid_f1_score_macro=0.3010" in log_line
    assert "f1_macro" not in log_line


def test_instance_iou_matching_counts_class_aware_matches():
    from metrics import InstanceIoUMatcher

    gt_inst = np.array([[1, 1, 0], [0, 2, 2]])
    gt_sem = np.array([[1, 1, 0], [0, 2, 2]])
    pred_sem = np.array([[1, 1, 0], [0, 2, 2]])
    matcher = InstanceIoUMatcher(num_classes=3, iou_threshold=0.5)
    matcher.update(pred_sem, gt_sem, gt_inst)
    out = matcher.compute()
    assert out["instance_iou_match/f1_micro"] > 0.99
    assert out["instance_iou_match/mean_iou"] > 0.99


def test_pfmb_loss_excludes_background_and_returns_scalar():
    from losses import EMAInverseClassFrequencyLoss

    base = torch.nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    loss_fn = EMAInverseClassFrequencyLoss(
        base,
        exclude_class=[0],
        num_classes=3,
        class_weighting=True,
    )
    pred = torch.randn(2, 3, 4, 4)
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    target[:, 1:3, 1:3] = 1
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_test_rows_use_label_names_and_exclude_background():
    from training.train import _classwise_test_rows

    metrics = {
        "accuracy_overall": 0.5,
        "f1_macro": 0.4,
        "f1_micro": 0.5,
        "dice_macro": 0.4,
        "dice_micro": 0.5,
        "iou_macro": 0.3,
        "iou_micro": 0.4,
        "foreground_iou": 0.2,
        "precision_macro": 0.6,
        "precision_micro": 0.7,
        "recall_macro": 0.8,
        "recall_micro": 0.9,
        "loss": 1.0,
        "class_1/support": 10,
        "class_1/f1": 0.25,
        "class_1/dice": 0.25,
        "class_1/iou": 0.15,
        "class_1/precision": 0.5,
        "class_1/recall": 0.2,
        "class_1/accuracy": 0.75,
    }
    rows = _classwise_test_rows(
        metrics,
        {0: "Background", 1: "B cells"},
        exclude_background=True,
    )
    assert all(row["class_name"] != "Background" for row in rows)
    assert rows[1]["class_name"] == "B cells"


def test_tune_binary_threshold_prefers_lower_cutoff_for_missed_positives():
    from metrics import predictions_from_threshold, tune_binary_threshold

    labels = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    positive_probs = np.array([0.10, 0.30, 0.80, 0.35, 0.05, 0.55], dtype=float)

    threshold, score = tune_binary_threshold(labels, positive_probs, metric="f1_macro")
    predictions = predictions_from_threshold(positive_probs, threshold)

    assert threshold < 0.5
    assert score > 0.0
    assert predictions[1] == 1
    assert predictions[3] == 1


def test_normalize_mil_features_l2_unit_vectors():
    from data.feature_bags import _normalize_mil_features

    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = _normalize_mil_features(x, "l2")
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.0]))


def test_normalize_mil_features_zscore_train():
    from data.feature_bags import _normalize_mil_features

    x = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    mean = torch.tensor([2.0, 4.0])
    std = torch.tensor([1.0, 1.0])
    out = _normalize_mil_features(x, "zscore_train", mean=mean, std=std)
    assert torch.allclose(out, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


def test_pathocell_grid_eval_keeps_full_volume_reference():
    """Regression: overwriting the volume array inside the tile loop only scores tile (0, 0)."""
    import numpy as np

    from data.tiling import (
        PFMB_TILE_PROTOCOL,
        extract_tile_arrays,
        plan_tiles,
        tile_top_left,
    )
    from metrics import get_instance_level_labels

    height = width = 448
    tile = 224
    volume = np.zeros((height, width, 3), dtype=np.uint8)
    semantic_hw = np.zeros((height, width), dtype=np.int64)
    instance_hw = np.zeros((height, width), dtype=np.int64)
    # Distinct instances in each quadrant so only correct tiling sees all four.
    instance_hw[32, 32] = 1
    instance_hw[32, 352] = 2
    instance_hw[352, 32] = 3
    instance_hw[352, 352] = 4
    for inst_id, (y, x) in enumerate([(32, 32), (32, 352), (352, 32), (352, 352)], start=1):
        semantic_hw[y, x] = inst_id

    label_dict = {0: "background", 1: "c1", 2: "c2", 3: "c3", 4: "c4"}
    _, tiles = plan_tiles(height, width, tile, protocol=PFMB_TILE_PROTOCOL)

    def count_instances() -> int:
        total = 0
        for tile_bounds in tiles:
            tile_image, semantic, instance = extract_tile_arrays(
                volume,
                semantic_hw,
                instance_hw,
                tile_bounds,
                tile,
                tile,
                protocol=PFMB_TILE_PROTOCOL,
            )
            assert tile_image.shape[:2] == (tile, tile)
            oracle = np.stack([(semantic == cls).astype(np.float32) for cls in label_dict])
            y_true, _ = get_instance_level_labels(oracle, semantic, instance, label_dict)
            total += len(y_true)
        return total

    assert count_instances() == 4
    assert not volume.any(), "tiling must not mutate the source volume"
