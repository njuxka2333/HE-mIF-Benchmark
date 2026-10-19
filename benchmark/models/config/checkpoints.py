"""Shim: checkpoint loading lives in ``utils.checkpoints``."""

from utils.checkpoints import *  # noqa: F403
from utils.checkpoints import (  # noqa: F401
    DEFAULT_CKPT_DIR,
    download_hub_checkpoint,
    download_url_checkpoint,
    hub_reference,
    resolve_model_checkpoint,
    resolve_path,
)
