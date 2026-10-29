"""WSI opening, path discovery, and feature artifact writers."""

from __future__ import annotations

import os
import time
from multiprocessing import Process
from pathlib import Path

import torch

from utils.file_utils import save_hdf5, save_patch_bag_h5


def open_wsi(path: str | Path):
    """Open a whole-slide image with the backend used by its file format."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"WSI does not exist: {path}")

    normalized = path.lower().replace("\\", "/")
    if normalized.endswith((".tif", ".tiff")) or "/t-cairem/" in normalized or "/tcairem/" in normalized:
        import openslide

        return openslide.OpenSlide(path)

    from Aslide import Slide

    return Slide(path)


def find_wsi_paths(root: str | Path, extensions: str) -> dict[str, str]:
    """Index WSI files recursively by slide id and accepted suffixes."""
    suffixes = {
        item.strip().lower().lstrip(".")
        for item in str(extensions).split(";")
        if item.strip()
    }
    paths = {}
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix.lower().lstrip(".") not in suffixes:
            continue
        paths[path.stem] = str(path)
    return paths


def _save_in_process(target, kwargs: dict) -> None:
    process = Process(target=target, kwargs=kwargs)
    process.start()
    process.join()
    if process.exitcode != 0:
        output = kwargs.get("output_path") or kwargs.get("path")
        raise RuntimeError(f"Saving {output} failed with exit code {process.exitcode}")


def _save_tensor(path: str, tensor: torch.Tensor) -> None:
    started = time.time()
    torch.save(tensor, path)
    print(f"Saved feature tensor to {path} in {time.time() - started:.1f}s")


def save_tensor_subprocess(path: str, tensor: torch.Tensor) -> None:
    _save_in_process(_save_tensor, {"path": path, "tensor": tensor})


def save_hdf5_subprocess(output_path, asset_dict, attr_dict=None) -> None:
    _save_in_process(
        save_hdf5,
        {"output_path": output_path, "asset_dict": asset_dict, "attr_dict": attr_dict, "mode": "w"},
    )


def save_patch_bag_subprocess(
    output_path,
    features,
    coords,
    masks=None,
    attr_dict=None,
    *,
    feature_dtype="float16",
    mask_dtype="uint8",
) -> None:
    _save_in_process(
        save_patch_bag_h5,
        {
            "output_path": output_path,
            "feature_obj": features,
            "coords": coords,
            "masks": masks,
            "attr_dict": attr_dict,
            "feature_dtype": feature_dtype,
            "mask_dtype": mask_dtype,
        },
    )
