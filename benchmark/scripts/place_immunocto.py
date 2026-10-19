#!/usr/bin/env python3
"""Re-place extracted Immunocto features onto the leakage-free partition.

Why this is a placement step and not a config edit: the linear-probe loader
takes split membership from the *directory* a feature file sits in, and the only
config-level control is ``exclude_case_ids``, which can remove a case from a
split but cannot move one into another. Moving CRC02/11/19/30 into the test
split therefore means re-placing files. Nothing is re-extracted -- the features
are hardlinked into a new root, so the cost is inodes, not GPU hours, and the
original layout stays untouched for comparison.

The script DISCOVERS the on-disk layout rather than assuming it, and refuses to
act if what it finds does not match the protocol. Run with --dry-run first and
read the plan it prints.

    python scripts/place_immunocto.py \\
        --src /data3/.../Pre_Features/linear_probe_immunocto_uni2_20x_s32/encoder_z4 \\
        --dry-run

See :mod:`data.immunocto` for why these four slides, and for the two
residual caveats that belong in the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.immunocto import (  # noqa: E402
    LEAKFREE_TEST, assignment, describe, split_of,
)

CASE_RE = re.compile(r"(CRC\d{2})", re.IGNORECASE)
SPLITS = ("train", "valid", "test")


def case_of(path: Path) -> str | None:
    """Specimen id for a feature file, from its name or any parent directory."""
    for part in (path.name, *[p.name for p in path.parents]):
        m = CASE_RE.search(part)
        if m:
            return m.group(1).upper()
    return None


# Priority when a case appears under several split directories. ``valid`` wins
# over ``train`` because the superseded layout left stale copies in ``train``
# for cases it excluded there by config.
SOURCE_PRIORITY = ("test", "valid", "train")


def plan_sources(found: dict[str, dict[str, list[Path]]]) -> dict[str, tuple[str, list[Path]]]:
    """One source split per case: {case_id: (split, files)}."""
    chosen: dict[str, tuple[str, list[Path]]] = {}
    for split in SOURCE_PRIORITY:
        for case, files in found.get(split, {}).items():
            chosen.setdefault(case, (split, files))
    return chosen


def discover(src: Path) -> dict[str, dict[str, list[Path]]]:
    """{split: {case_id: [files]}} for the existing layout.

    A case may appear under more than one split directory: the superseded
    partition kept CRC29-CRC32 physically in ``train/`` while excluding them
    from training through ``exclude_case_ids_by_split``, and also placed them
    in ``valid/``. Placing every copy would emit that case twice into its new
    split and double-weight it in the metrics, so :func:`plan_sources` keeps
    exactly one source per case.
    """
    found: dict[str, dict[str, list[Path]]] = {s: defaultdict(list) for s in SPLITS}
    for split in SPLITS:
        split_dir = src / split
        if not split_dir.is_dir():
            raise SystemExit(
                f"expected split directory {split_dir} not found.\n"
                f"  {src} holds: {sorted(p.name for p in src.iterdir())[:12]}\n"
                "  This script only handles the protocol's train/valid/test layout."
            )
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file():
                continue
            case = case_of(path.relative_to(split_dir)) or case_of(path)
            if case is None:
                raise SystemExit(
                    f"cannot read a CRC## specimen id from {path}.\n"
                    "  Per-case file or directory names are required to re-split; "
                    "if the features are stored as one h5 per split they must be "
                    "re-extracted instead."
                )
            found[split][case].append(path)
    return {s: dict(v) for s, v in found.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="existing <run>/encoder_z4 or output_layers root")
    ap.add_argument("--dst", type=Path, default=None, help="default: <src parent>/<layer>_leakfree")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst = (args.dst or src.parent / f"{src.name}_leakfree").expanduser().resolve()
    print(describe())
    print(f"src {src}\ndst {dst}\n")

    found = discover(src)
    cases = {c for split in found.values() for c in split}
    print("current layout: " + "  ".join(f"{s}={len(found[s])} cases" for s in SPLITS))
    missing = set(LEAKFREE_TEST) - cases
    if missing:
        raise SystemExit(f"test slides absent from the extracted features: {sorted(missing)}")

    sources = plan_sources(found)
    plan: dict[str, list[str]] = {s: [] for s in SPLITS}
    moves = 0
    for case, (split, _files) in sorted(sources.items()):
        target = split_of(case)
        plan[target].append(case)
        if target != split:
            moves += 1
    duplicated = {c: sorted(s for s in found if c in found[s]) for c in sources
                  if sum(c in found[s] for s in SPLITS) > 1}
    if duplicated:
        print("\ncases present under more than one split directory "
              "(one source kept each, the rest ignored):")
        for case, splits in sorted(duplicated.items()):
            print(f"  {case}: {', '.join(splits)} -> keeping {sources[case][0]}")
    for s in SPLITS:
        print(f"  new {s:<6} {len(plan[s]):>2} cases: {', '.join(sorted(plan[s]))}")
    print(f"\n{moves} of {len(cases)} cases change split")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return

    n = 0
    for case, (split, files) in sorted(sources.items()):
        if True:
            for f in files:
                out = dst / split_of(case) / f.relative_to(src / split)
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists():
                    continue
                try:
                    os.link(f, out)
                except OSError:
                    import shutil
                    shutil.copy2(f, out)
                n += 1
    (dst / "leakfree_partition.json").write_text(json.dumps(
        {"reason": "MIPHEI-ViT trained on 36/40 Orion slides; see data/immunocto.py",
         "assignment": {k: list(v) for k, v in assignment().items()},
         "source": str(src), "files_placed": n}, indent=2))
    print(f"\nplaced {n} files -> {dst}")


if __name__ == "__main__":
    main()
