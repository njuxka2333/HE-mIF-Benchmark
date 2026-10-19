"""Pick a .pyc whose magic matches the running interpreter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EXPECTED_MAGIC = importlib.util.MAGIC_NUMBER


def pyc_magic_matches(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == _EXPECTED_MAGIC


def pick_pyc(module_stem: str, here: Path | None = None, *, min_size: int = 2500) -> Path:
    root = here or Path(__file__).resolve().parent
    # When a shim is executed from __pycache__/foo.cpython-311.pyc, callers sometimes
    # pass stem "foo.cpython-311". Normalize so we still find foo.* under _orig_pyc.
    if ".cpython-" in module_stem:
        module_stem = module_stem.split(".cpython-", 1)[0]
    # If here accidentally points at __pycache__, step up to the package dir.
    if root.name == "__pycache__":
        root = root.parent
    major, minor = sys.version_info[:2]
    tag = f"cpython-{major}{minor}"
    candidates = [
        root / "_orig_pyc" / f"{module_stem}.pyc",
        root / "__pycache__" / f"{module_stem}.{tag}.pyc",
        root / "__pycache__" / f"{module_stem}.cpython-311.pyc",
        root / "__pycache__" / f"{module_stem}.cpython-310.pyc",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if path.stat().st_size < min_size:
            continue
        if not pyc_magic_matches(path):
            continue
        return path
    raise ImportError(
        f"No usable bytecode for {module_stem!r} under {root} "
        f"(Python {major}.{minor}); candidates={[str(p) for p in candidates]}"
    )
