"""Offline bag training: linear probe and survival.

The training loop itself still lives in archived bytecode
(``_orig_pyc/offline_train.pyc``). Patches below keep SCC on AdamW, allow the
Shazam survival head, and attach PCA hooks — those are the only source-level
differences from the archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

from training._bytecode_loader import install_bytecode_shim

install_bytecode_shim(
    "offline_train",
    Path(__file__).resolve().parent,
    globals(),
    min_bytecode_size=5000,
)
_mod = sys.modules["_bytecode_offline_train"]

from metrics import linear_probe_metrics as linear_probe_metrics  # noqa: E402,F811

_mod.linear_probe_metrics = linear_probe_metrics

_BYTECODE_VALIDATE_TASK_CONFIG = _mod._validate_task_config


def _validate_task_config(config):
    try:
        return _BYTECODE_VALIDATE_TASK_CONFIG(config)
    except ValueError as exc:
        text = str(exc)
        if (
            "expects head='survival_attention_mil'" in text
            and "got head='shazam_survival_abmil'" in text
        ):
            return None
        raise


_mod._validate_task_config = _validate_task_config

from training.linear_probe_adamw import run_linear_probe as _run_linear_probe_adamw  # noqa: E402
from training.pca_features import install_pca_hook  # noqa: E402

install_pca_hook(_mod)


def run_linear_probe(cfg):
    """SCC always uses AdamW; bytecode LBFGS ``run_linear_probe`` is not called."""
    return _run_linear_probe_adamw(cfg)


def _fit_linear_probe(*_args, **_kwargs):
    raise RuntimeError("LBFGS linear probe has been removed; use AdamW + max_epochs.")


_mod.run_linear_probe = run_linear_probe
_mod._fit_linear_probe = _fit_linear_probe
globals()["_validate_task_config"] = _validate_task_config
globals()["run_linear_probe"] = run_linear_probe
globals()["_fit_linear_probe"] = _fit_linear_probe

if __name__ == "__main__":
    main()
