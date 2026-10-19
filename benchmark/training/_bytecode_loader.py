"""Shared helpers for loading archived training bytecode on 3.11 or hemif 3.10."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def install_bytecode_shim(
    stem: str,
    here: Path,
    namespace: dict[str, Any],
    *,
    min_bytecode_size: int = 2500,
) -> None:
    """Load archived ``training/<stem>`` implementation into ``namespace``."""
    major, minor = sys.version_info[:2]
    modname = f"_bytecode_{stem}"
    skip = {
        "__name__",
        "__file__",
        "__cached__",
        "__spec__",
        "__loader__",
        "__package__",
        "__builtins__",
        "__doc__",
    }

    if (major, minor) >= (3, 11):
        from training._bytecode_pick import pick_pyc

        pyc = pick_pyc(stem, here, min_size=min_bytecode_size)
        spec = importlib.util.spec_from_file_location(modname, pyc)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {pyc}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        exported = {k: v for k, v in vars(mod).items() if k not in skip}
    else:
        dill_path = here / "_dill" / f"{stem}_py311.dill"
        if not dill_path.is_file() or dill_path.stat().st_size < 200:
            raise ImportError(
                f"training.{stem} requires Python 3.11 bytecode "
                f"(or a prebuilt cache at {dill_path})."
            )
        import dill

        # dill may reference ``modname`` while unpickling nested objects.
        mod = types.ModuleType(modname)
        sys.modules[modname] = mod
        payload = dill.load(dill_path.open("rb"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict in {dill_path}, got {type(payload)}")
        mod.__dict__.update(payload)
        exported = {k: v for k, v in payload.items() if k not in skip}

    namespace.update(exported)
