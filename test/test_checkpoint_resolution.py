"""Checkpoint resolver behavior for local and Hub-backed model configs."""

from __future__ import annotations

from models.schema import HubCheckpointConfig, ModelConfig


def test_local_checkpoint_takes_priority_over_hub(tmp_path, monkeypatch):
    import utils.checkpoints as checkpoints

    local = tmp_path / "local_weights.pth"
    local.write_bytes(b"weights")
    cfg = ModelConfig(
        name="example",
        checkpoint_path=str(local),
        hub=HubCheckpointConfig(repo_id="org/model", filename="remote.pth"),
    )

    def fail_if_called(_hub):
        raise AssertionError("Hub download should not run when local checkpoint exists")

    monkeypatch.setattr(checkpoints, "download_hub_checkpoint", fail_if_called)

    assert checkpoints.resolve_model_checkpoint(cfg) == str(local.resolve())


def test_missing_local_checkpoint_falls_back_to_hub(tmp_path, monkeypatch):
    import utils.checkpoints as checkpoints

    remote = tmp_path / "remote_weights.pth"
    remote.write_bytes(b"weights")
    cfg = ModelConfig(
        name="example",
        checkpoint_path=str(tmp_path / "missing.pth"),
        hub=HubCheckpointConfig(repo_id="org/model", filename="remote.pth"),
    )

    monkeypatch.setattr(
        checkpoints,
        "download_hub_checkpoint",
        lambda _hub: str(remote.resolve()),
    )

    assert checkpoints.resolve_model_checkpoint(cfg) == str(remote.resolve())


def test_default_checkpoint_root_is_repo_checkpoints():
    from utils.paths import REPO_ROOT, UNI2_CHECKPOINT, checkpoint_root, checkpoint_search_dirs

    assert checkpoint_root() == (REPO_ROOT / "checkpoints").resolve()
    assert checkpoint_search_dirs() == [checkpoint_root()]
    assert UNI2_CHECKPOINT == checkpoint_root() / "uni2_pytorch_model.bin"
