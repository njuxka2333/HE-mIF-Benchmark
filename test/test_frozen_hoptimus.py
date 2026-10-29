"""Frozen H-optimus-0 extract runtime (MIPHEI minus LoRA)."""

from __future__ import annotations

import os

from feature_extraction.extract import _apply_extract_runtime
from feature_extraction.frozen_hoptimus import frozen_hoptimus_requested


def test_extract_env_sets_lora_skip(monkeypatch):
    monkeypatch.delenv("MIPHEI_SKIP_GENERATOR_CKPT", raising=False)
    monkeypatch.delenv("MIPHEI_USE_LORA", raising=False)
    called = {}

    def _fake_install():
        called["yes"] = True
        os.environ["MIPHEI_USE_LORA"] = "0"

    monkeypatch.setattr("feature_extraction.frozen_hoptimus.configure_frozen_hoptimus", _fake_install)
    _apply_extract_runtime({"env": {"MIPHEI_SKIP_GENERATOR_CKPT": "1", "MIPHEI_USE_LORA": "0"}})
    assert os.environ["MIPHEI_SKIP_GENERATOR_CKPT"] == "1"
    assert frozen_hoptimus_requested()
    assert called["yes"]
    assert os.environ["MIPHEI_USE_LORA"] == "0"


def test_apply_extract_env_from_config_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("MIPHEI_SKIP_GENERATOR_CKPT", raising=False)
    monkeypatch.delenv("MIPHEI_USE_LORA", raising=False)
    called = {}

    def _fake_install():
        called["yes"] = True
        os.environ["MIPHEI_USE_LORA"] = "0"

    monkeypatch.setattr("feature_extraction.frozen_hoptimus.configure_frozen_hoptimus", _fake_install)
    path = tmp_path / "hoptimus0.yaml"
    path.write_text(
        "extract:\n  env:\n    MIPHEI_SKIP_GENERATOR_CKPT: '1'\n    MIPHEI_USE_LORA: '0'\n",
        encoding="utf-8",
    )
    from feature_extraction.frozen_hoptimus import apply_extract_env_from_config

    apply_extract_env_from_config(path)
    assert os.environ["MIPHEI_SKIP_GENERATOR_CKPT"] == "1"
    assert os.environ["MIPHEI_USE_LORA"] == "0"
    assert called["yes"]


def test_apply_model_env_does_not_reenable_lora(monkeypatch):
    monkeypatch.setenv("MIPHEI_SKIP_GENERATOR_CKPT", "1")
    monkeypatch.setenv("MIPHEI_USE_LORA", "0")
    from models.registry import apply_model_env, get_model_config

    cfg = get_model_config("mipheivit")
    apply_model_env(cfg)
    assert os.environ["MIPHEI_USE_LORA"] == "0"


def test_extract_env_noop_without_skip(monkeypatch):
    monkeypatch.delenv("MIPHEI_SKIP_GENERATOR_CKPT", raising=False)
    called = {}
    monkeypatch.setattr(
        "feature_extraction.frozen_hoptimus.configure_frozen_hoptimus",
        lambda: called.setdefault("yes", True),
    )
    _apply_extract_runtime({})
    assert not called
    assert not frozen_hoptimus_requested()
