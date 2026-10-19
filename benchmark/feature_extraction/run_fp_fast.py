"""Shim: source missing; load best available bytecode for this Python."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
_NAME = Path(__file__).resolve().stem
_MAJOR, _MINOR = sys.version_info[:2]
_TAG = f"cpython-{_MAJOR}{_MINOR}"

_CANDIDATES = [
    _HERE / "_orig_pyc" / f"{_NAME}.pyc",
    _HERE / "__pycache__" / f"{_NAME}.{_TAG}.pyc",
    _HERE / "__pycache__" / f"{_NAME}.cpython-311.pyc",
    _HERE / "__pycache__" / f"{_NAME}.cpython-310.pyc",
]

def _pick() -> Path:
    for path in _CANDIDATES:
        if path.is_file() and path.stat().st_size >= 10000:
            return path
    raise ImportError(
        f"No usable bytecode for {_NAME!r} under {_HERE} "
        f"(Python {_MAJOR}.{_MINOR}); candidates={[str(p) for p in _CANDIDATES]}"
    )

_PYC = _pick()
_MODNAME = f"_bytecode_{_NAME}"
_spec = importlib.util.spec_from_file_location(_MODNAME, _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load {_PYC}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MODNAME] = _mod
_spec.loader.exec_module(_mod)
_SKIP = {
    "__name__", "__file__", "__cached__", "__spec__", "__loader__",
    "__package__", "__builtins__", "__doc__",
}
globals().update({k: v for k, v in vars(_mod).items() if k not in _SKIP})
