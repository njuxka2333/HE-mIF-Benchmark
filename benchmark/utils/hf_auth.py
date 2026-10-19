"""Hugging Face authentication (Pathology-Foundation-Model-Benchmark pattern)."""

from __future__ import annotations

import os

from utils.paths import PREPATH_ROOT, REPO_ROOT

_HF_LOGIN_DONE = False


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for env_path in (REPO_ROOT / ".env", PREPATH_ROOT / ".env"):
        if env_path.is_file():
            load_dotenv(env_path)


def login_huggingface() -> None:
    """Load ``HF_TOKEN`` from ``.env`` and authenticate with Hugging Face Hub."""
    global _HF_LOGIN_DONE
    if _HF_LOGIN_DONE:
        return
    _load_env()
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token)
    _HF_LOGIN_DONE = True


def hf_token() -> str | None:
    """Return ``HF_TOKEN`` from the environment (after loading ``.env``)."""
    _load_env()
    return os.getenv("HF_TOKEN")
