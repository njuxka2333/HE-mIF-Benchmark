#!/usr/bin/env python3
"""Run pytest for the nested HE-mIF-Benchmark repo.

Examples:
  python test/run_tests.py                  # full suite (includes weight/integration tests)
  python test/run_tests.py --no-weights     # everything except checkpoint load / E2E extraction
  CUDA_VISIBLE_DEVICES=2 python test/run_tests.py --weights   # model + E2E tests on one GPU
  python test/run_tests.py -m integration   # forward to pytest markers
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))


def main() -> int:
    argv = list(sys.argv[1:])
    pytest_args = [str(TESTS_DIR), "-v"]

    if "--no-weights" in argv:
        argv.remove("--no-weights")
        pytest_args.extend(["-m", "not requires_weights"])
    elif "--weights" in argv:
        argv.remove("--weights")
        pytest_args.extend(["-m", "requires_weights"])

    pytest_args.extend(argv)
    print(f"Using interpreter: {sys.executable}")
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
