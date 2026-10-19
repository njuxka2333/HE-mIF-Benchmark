"""Shim: source missing; load archived bytecode."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
_PYC = _HERE / "_orig_pyc" / "hemmit_inference.pyc"
_spec = importlib.util.spec_from_file_location("_bytecode_hemmit_inference", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load {_PYC}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_bytecode_hemmit_inference"] = _mod
_spec.loader.exec_module(_mod)
_SKIP = {
    "__name__", "__file__", "__cached__", "__spec__", "__loader__",
    "__package__", "__builtins__", "__doc__",
}
globals().update({k: v for k, v in vars(_mod).items() if k not in _SKIP})
