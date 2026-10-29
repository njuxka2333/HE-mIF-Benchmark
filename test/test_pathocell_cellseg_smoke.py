"""Smoke tests for PFMB-aligned PathoCell online cell semantic segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.loader import get_dataset_config, load_split_samples
from data.pathocell.sampler import HDFTileBatchSampler
from training.augmentations import InputNormalization, build_augmenter, find_input_normalization
from heads import build_segmentation_head, forward_segmentation_head
from training.online_segmentation_train import (
    _assert_pathocell_segmentation_head,
    load_online_eval_bundle,
    maybe_augment,
    resolve_num_classes,
    train_one_online_step,
)

PATHOCELL_ONLINE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/downstream/cell_semantic_segmentation/pathocell/encoder/uni2.yaml"
)


def _mini_exp():
    return type(
        "Exp",
        (),
        {
            "target_patch_size": 224,
            "target_mpp": 0.5,
            "use_mpp": True,
            "target_features": ["z1", "z2", "z3", "z4"],
            "dataset_config": get_dataset_config("pathocell"),
        },
    )()


class _MockEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        return {
            name: torch.randn(batch_size, 1536, 16, 16, device=images.device, dtype=images.dtype)
            for name in ("z1", "z2", "z3", "z4")
        }


def test_pathocell_online_config_pfmb_settings():
    assert PATHOCELL_ONLINE_CONFIG.is_file()
    raw = yaml.safe_load(PATHOCELL_ONLINE_CONFIG.read_text(encoding="utf-8"))
    assert raw["evaluation"]["head"] == "unetr"
    assert raw["head"]["use_z0"] is True
    assert raw["training"]["global_batch_size"] == 128
    assert raw["data"]["online_sampling"]["global_batch_size"] == 128
    assert raw["training"]["gradient_accumulation"] == 1
    assert raw["training"]["training_steps"] == 20000
    assert raw["training"]["val_interval"] == 200
    assert raw["training"]["early_stopping"] == 1200
    assert raw["experiment"]["pathocell_tiling"] == "pfmb"
    assert raw["data"]["pathocell_tiling"] == "pfmb"
    assert raw["data"]["online_sampling"]["prefetch_batches"] == 2
    assert raw["data"]["online_sampling"]["volume_cache_size"] == 8
    assert raw["training"]["uniform_class_sampling"] is True
    assert raw["training"]["primary_metric"] == "f1_score_macro"
    assert "RandomHorizontalFlip" in raw["training"]["augmentations"]


def test_pathocell_online_bundle_loads():
    exp, cfg = load_online_eval_bundle(PATHOCELL_ONLINE_CONFIG)
    assert cfg.task == "cell_semantic_segmentation"
    assert cfg.head == "unetr"
    _assert_pathocell_segmentation_head(cfg)
    assert exp.pathocell_tiling == "pfmb"
    assert exp.model == "uni2"
    assert resolve_num_classes(cfg, exp) >= 2


def test_pathocell_pfmb_augmenter_from_config():
    raw = yaml.safe_load(PATHOCELL_ONLINE_CONFIG.read_text(encoding="utf-8"))
    augmenter = build_augmenter(raw["training"])
    assert augmenter is not None
    images = torch.rand(2, 3, 224, 224)
    masks = torch.randint(0, 5, (2, 224, 224))
    instances = torch.randint(0, 20, (2, 224, 224))
    aug_images, aug_masks, aug_instances = augmenter(images, masks, instances)
    assert aug_images.shape == images.shape
    assert aug_masks.shape == masks.shape
    assert aug_instances.shape == instances.shape
    assert aug_masks.dtype == masks.dtype
    assert aug_instances.dtype == instances.dtype


def test_pfmb_augmentation_runs_before_model_normalization():
    class _CaptureAugmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, images, masks, instances):
            self.seen = images.detach().clone()
            return images, masks, instances

    normalization = InputNormalization(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    raw = torch.rand(2, 3, 32, 32)
    normalized = normalization.encode(raw)
    masks = torch.zeros(2, 32, 32, dtype=torch.long)
    instances = torch.ones(2, 32, 32, dtype=torch.long)
    augmenter = _CaptureAugmenter()

    output, _, _ = maybe_augment(
        augmenter,
        normalized,
        masks,
        instances,
        input_normalization=normalization,
    )

    assert augmenter.seen is not None
    assert torch.allclose(augmenter.seen, raw, atol=1e-6)
    assert torch.allclose(output, normalized, atol=1e-6)


def test_model_normalization_is_discovered_from_compose():
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.4, 0.3), (0.2, 0.3, 0.4)),
        ]
    )
    normalization = find_input_normalization(transform)
    assert normalization is not None
    assert normalization.mean == (0.5, 0.4, 0.3)
    assert normalization.std == (0.2, 0.3, 0.4)


def test_pathocell_uniform_class_sampler():
    cfg = get_dataset_config("pathocell")
    samples = load_split_samples(cfg, split_filter="train", max_samples=2)
    if not samples:
        return
    sampler = HDFTileBatchSampler(
        _mini_exp(),
        samples,
        image_transform=None,
        batch_size=2,
        seed=0,
        uniform_class_sampling=True,
        num_classes=15,
        exclude_classes=[0],
        sample_excluded_classes=False,
    )
    assert sampler._tile_weights is not None
    assert sampler._tile_weights.shape[0] == len(sampler.tile_refs)
    batch = sampler.next_batch()
    assert batch["image"].shape[0] == 2
    assert batch["semantic_mask"].shape[0] == 2
    assert batch["instance_mask"].shape == batch["semantic_mask"].shape
    sampler.close()


def test_pathocell_online_train_step_is_one_optimizer_update():
    device = torch.device("cpu")
    decoder = build_segmentation_head(
        "unetr",
        feature_layers=["z1", "z2", "z3", "z4"],
        in_channels=[1536, 1536, 1536, 1536],
        num_classes=4,
        head_config={"use_z0": True, "drop_rate": 0.0},
    ).to(device)
    encoder = _MockEncoder().to(device)
    raw = yaml.safe_load(PATHOCELL_ONLINE_CONFIG.read_text(encoding="utf-8"))
    augmenter = build_augmenter(raw["training"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    batch = {
        "image": torch.rand(2, 3, 224, 224),
        "semantic_mask": torch.randint(0, 4, (2, 224, 224)),
        "instance_mask": torch.randint(1, 10, (2, 224, 224)),
    }
    images = batch["image"].to(device)
    masks = batch["semantic_mask"].to(device)
    instances = batch["instance_mask"].to(device)
    aug_images, aug_masks, aug_instances = maybe_augment(augmenter, images, masks, instances)
    assert aug_images.shape == images.shape
    assert aug_masks.shape == masks.shape
    assert aug_instances.shape == instances.shape

    optimizer.zero_grad(set_to_none=True)
    scheduler_before = scheduler.last_epoch
    loss_1 = train_one_online_step(
        decoder,
        encoder,
        batch,
        criterion,
        optimizer,
        scheduler,
        scaler,
        augmenter,
        device=device,
        use_amp=False,
        grad_clip_norm=2.0,
    )
    assert loss_1 == loss_1
    assert scheduler.last_epoch == scheduler_before + 1
    assert all(param.grad is None for param in decoder.parameters())
    logits = forward_segmentation_head(
        decoder,
        encoder(batch["image"].to(device)),
        batch["semantic_mask"].to(device),
        images=batch["image"].to(device),
    )
    assert logits.shape == (2, 4, 224, 224)
