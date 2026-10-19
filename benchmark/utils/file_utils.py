import json
import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

PATCH_BAG_SCHEMA_VERSION = "patch_bag_v1"
PATCH_BAG_STAGE_MASKS_ONLY = "masks_only"
PATCH_BAG_STAGE_COMPLETE = "complete"


def cast_features_for_storage(feature_obj: Any, feature_dtype: str) -> Any:
    """Cast torch feature tensors to the on-disk dtype (float16 or float32)."""
    torch_dtype = torch.float16 if str(feature_dtype) == "float16" else torch.float32
    if isinstance(feature_obj, dict):
        return {key: value.to(dtype=torch_dtype) for key, value in feature_obj.items()}
    return feature_obj.to(dtype=torch_dtype)


def cast_masks_for_storage(masks: Any, mask_dtype: str) -> np.ndarray | None:
    if masks is None:
        return None
    np_dtype = np.uint8 if str(mask_dtype) == "uint8" else np.int32
    return np.asarray(masks).astype(np_dtype, copy=False)


def save_pkl(filename, save_object):
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename):
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file

def collate_features(batch):
    img = torch.stack([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    assert len(img.shape) == 4, "img shape is wrong, please check"
    if len(batch[0]) > 2 and batch[0][2] is not None:
        masks = np.stack([item[2] for item in batch], axis=0)
        return [img, coords, masks]
    return [img, coords]


def _to_numpy_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def is_patch_bag_h5(path: str | Any) -> bool:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") == PATCH_BAG_SCHEMA_VERSION:
            return True
        return "features" in handle and isinstance(handle["features"], h5py.Group)


def is_masks_only_patch_bag_h5(path: str | Path) -> bool:
    """True when coords/masks are stored and features can be appended later."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != PATCH_BAG_SCHEMA_VERSION:
                return False
            if "coords" not in handle or "masks" not in handle:
                return False
            n_patches = int(handle["coords"].shape[0])
            if n_patches <= 0:
                return False
            if int(handle["masks"].shape[0]) != n_patches:
                return False
            if "features" in handle and len(handle["features"].keys()) > 0:
                return False
            return handle.attrs.get("bag_stage", PATCH_BAG_STAGE_MASKS_ONLY) == PATCH_BAG_STAGE_MASKS_ONLY
    except OSError:
        return False


def is_complete_patch_bag_h5(path: str | Path) -> bool:
    """True when a patch bag finished writing and is readable for training/resume."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != PATCH_BAG_SCHEMA_VERSION:
                return False
            if "coords" not in handle:
                return False
            n_patches = int(handle["coords"].shape[0])
            if n_patches <= 0:
                return False
            if "features" not in handle:
                return False
            features = handle["features"]
            if isinstance(features, h5py.Group):
                if len(features.keys()) == 0:
                    return False
                for key in features.keys():
                    if int(features[key].shape[0]) != n_patches:
                        return False
            else:
                if int(features.shape[0]) != n_patches:
                    return False
            if "masks" in handle and int(handle["masks"].shape[0]) != n_patches:
                return False
        return True
    except OSError:
        return False


def patch_bag_part_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.name}.part")


def save_patch_bag_h5(
    output_path,
    feature_obj,
    coords,
    masks=None,
    *,
    attr_dict: dict[str, Any] | None = None,
    feature_dtype: np.dtype | str = np.float16,
    mask_dtype: np.dtype | str = np.uint8,
    patch_uids: np.ndarray | None = None,
    atomic: bool = True,
) -> str:
    """Write one sample bag with patch-indexed features stored inside HDF5."""
    output_path = Path(output_path)
    feat_dtype = np.dtype(feature_dtype)
    mask_storage_dtype = np.dtype(mask_dtype)
    coords_arr = _to_numpy_array(coords).astype(np.int32, copy=False)
    target_path = patch_bag_part_path(output_path) if atomic else output_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if atomic and target_path.is_file():
        target_path.unlink()

    with h5py.File(target_path, "w") as handle:
        handle.create_dataset(
            "coords",
            data=coords_arr,
            chunks=(1, 2),
            dtype=np.int32,
        )
        if masks is not None:
            masks_arr = _to_numpy_array(masks).astype(mask_storage_dtype, copy=False)
            handle.create_dataset(
                "masks",
                data=masks_arr,
                chunks=(1,) + masks_arr.shape[1:],
                dtype=mask_storage_dtype,
            )
        if patch_uids is not None:
            uid_arr = _to_numpy_array(patch_uids).astype(np.int64, copy=False)
            handle.create_dataset(
                "patch_uids",
                data=uid_arr,
                chunks=(1,),
                dtype=np.int64,
            )

        feature_group = handle.create_group("features")
        layer_names: list[str] = []
        if isinstance(feature_obj, dict):
            for name, tensor in feature_obj.items():
                arr = _to_numpy_array(tensor).astype(feat_dtype, copy=False)
                feature_group.create_dataset(
                    name,
                    data=arr,
                    chunks=(1,) + arr.shape[1:],
                    dtype=feat_dtype,
                )
                layer_names.append(str(name))
        else:
            arr = _to_numpy_array(feature_obj).astype(feat_dtype, copy=False)
            feature_group.create_dataset(
                "features",
                data=arr,
                chunks=(1,) + arr.shape[1:],
                dtype=feat_dtype,
            )
            layer_names = ["features"]

        handle.attrs["schema_version"] = PATCH_BAG_SCHEMA_VERSION
        handle.attrs["bag_stage"] = PATCH_BAG_STAGE_COMPLETE
        handle.attrs["feature_layers"] = json.dumps(layer_names)
        handle.attrs["feature_dtype"] = str(feat_dtype)
        if masks is not None:
            handle.attrs["mask_dtype"] = str(mask_storage_dtype)
        if attr_dict:
            for key, value in attr_dict.items():
                handle.attrs[key] = value

    if atomic:
        os.replace(target_path, output_path)
        final_path = output_path
    else:
        final_path = target_path

    print("Successfully saved patch bag:", final_path)
    return str(final_path)


class MaskOnlyPatchBagWriter:
    """Incrementally write coords/masks/patch_uids; features are appended later."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        patch_size: int,
        mask_dtype: np.dtype | str = np.uint8,
        attr_dict: dict[str, Any] | None = None,
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._patch_size = int(patch_size)
        mask_storage_dtype = np.dtype(mask_dtype)
        self._file = h5py.File(self._path, "w")
        self._count = 0
        self._file.create_dataset(
            "coords",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32,
            chunks=(1, 2),
        )
        self._file.create_dataset(
            "masks",
            shape=(0, self._patch_size, self._patch_size),
            maxshape=(None, self._patch_size, self._patch_size),
            dtype=mask_storage_dtype,
            chunks=(1, self._patch_size, self._patch_size),
        )
        self._file.create_dataset(
            "patch_uids",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(1,),
        )
        self._file.attrs["schema_version"] = PATCH_BAG_SCHEMA_VERSION
        self._file.attrs["bag_stage"] = PATCH_BAG_STAGE_MASKS_ONLY
        self._file.attrs["mask_dtype"] = str(mask_storage_dtype)
        if attr_dict:
            for key, value in attr_dict.items():
                self._file.attrs[key] = value

    def append(self, x: int, y: int, patch_uid: int, mask: np.ndarray) -> None:
        idx = self._count
        self._count += 1
        mask_arr = _to_numpy_array(mask).astype(self._file["masks"].dtype, copy=False)
        if mask_arr.shape != (self._patch_size, self._patch_size):
            raise ValueError(
                f"mask shape {mask_arr.shape} != ({self._patch_size}, {self._patch_size})"
            )
        for name, row in (
            ("coords", np.array([[int(x), int(y)]], dtype=np.int32)),
            ("masks", mask_arr.reshape(1, self._patch_size, self._patch_size)),
            ("patch_uids", np.array([int(patch_uid)], dtype=np.int64)),
        ):
            dataset = self._file[name]
            dataset.resize((self._count,) + dataset.shape[1:])
            dataset[idx] = row[0]

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

    def __enter__(self) -> "MaskOnlyPatchBagWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PatchBagIncrementalWriter:
    """Incrementally write coords/masks/patch_uids/features into one patch bag h5."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        patch_size: int,
        mask_dtype: np.dtype | str = np.uint8,
        feature_dtype: np.dtype | str = np.float16,
        attr_dict: dict[str, Any] | None = None,
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._patch_size = int(patch_size)
        self._mask_dtype = np.dtype(mask_dtype)
        self._feature_dtype = np.dtype(feature_dtype)
        self._file = h5py.File(self._path, "w")
        self._count = 0
        self._feature_layers: dict[str, Any] = {}
        self._file.create_dataset(
            "coords",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32,
            chunks=(1, 2),
        )
        self._file.create_dataset(
            "masks",
            shape=(0, self._patch_size, self._patch_size),
            maxshape=(None, self._patch_size, self._patch_size),
            dtype=self._mask_dtype,
            chunks=(1, self._patch_size, self._patch_size),
        )
        self._file.create_dataset(
            "patch_uids",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(1,),
        )
        self._file.attrs["schema_version"] = PATCH_BAG_SCHEMA_VERSION
        self._file.attrs["bag_stage"] = PATCH_BAG_STAGE_MASKS_ONLY
        self._file.attrs["mask_dtype"] = str(self._mask_dtype)
        self._file.attrs["feature_dtype"] = str(self._feature_dtype)
        if attr_dict:
            for key, value in attr_dict.items():
                self._file.attrs[key] = value

    def append(
        self,
        x: int,
        y: int,
        patch_uid: int,
        mask: np.ndarray,
        feature_layers: dict[str, Any],
    ) -> None:
        idx = self._count
        self._count += 1
        mask_arr = _to_numpy_array(mask).astype(self._mask_dtype, copy=False)
        if mask_arr.shape != (self._patch_size, self._patch_size):
            raise ValueError(
                f"mask shape {mask_arr.shape} != ({self._patch_size}, {self._patch_size})"
            )
        for name, row in (
            ("coords", np.array([[int(x), int(y)]], dtype=np.int32)),
            ("masks", mask_arr.reshape(1, self._patch_size, self._patch_size)),
            ("patch_uids", np.array([int(patch_uid)], dtype=np.int64)),
        ):
            dataset = self._file[name]
            dataset.resize((self._count,) + dataset.shape[1:])
            dataset[idx] = row[0]

        if not self._feature_layers:
            group = self._file.create_group("features")
            for name, tensor in feature_layers.items():
                arr = _to_numpy_array(tensor).astype(self._feature_dtype, copy=False)
                if arr.ndim == 0:
                    raise ValueError(f"feature layer {name} must be at least 1D")
                feat_shape = tuple(arr.shape)
                chunk_shape = (1,) + feat_shape
                dataset = group.create_dataset(
                    name,
                    shape=(1,) + feat_shape,
                    maxshape=(None,) + feat_shape,
                    dtype=self._feature_dtype,
                    chunks=chunk_shape,
                )
                dataset[0] = arr
                self._feature_layers[name] = dataset
            self._file.attrs["feature_layers"] = json.dumps(list(self._feature_layers.keys()))
            self._file.attrs["bag_stage"] = PATCH_BAG_STAGE_COMPLETE
        else:
            if set(feature_layers.keys()) != set(self._feature_layers.keys()):
                raise ValueError("inconsistent feature layer names across append calls")
            for name, tensor in feature_layers.items():
                arr = _to_numpy_array(tensor).astype(self._feature_dtype, copy=False)
                dataset = self._feature_layers[name]
                dataset.resize((self._count,) + dataset.shape[1:])
                dataset[idx] = arr

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

    @classmethod
    def open_existing(cls, output_path: str | Path) -> "PatchBagIncrementalWriter":
        """Resume incremental writing into an existing patch bag (coords/masks/features)."""
        self = cls.__new__(cls)
        self._path = Path(output_path)
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        self._file = h5py.File(self._path, "a")
        if "coords" not in self._file or "masks" not in self._file:
            raise ValueError(f"{self._path} is not a resumable patch bag")
        self._count = int(self._file["coords"].shape[0])
        self._patch_size = int(self._file["masks"].shape[1])
        self._mask_dtype = self._file["masks"].dtype
        self._feature_dtype = np.dtype(str(self._file.attrs.get("feature_dtype", "float16")))
        self._feature_layers = {}
        if "features" in self._file and isinstance(self._file["features"], h5py.Group):
            for name in self._file["features"].keys():
                self._feature_layers[name] = self._file["features"][name]
        return self

    def __enter__(self) -> "PatchBagIncrementalWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def save_masks_patch_bag_h5(
    output_path: str | Path,
    coords: np.ndarray,
    masks: np.ndarray,
    patch_uids: np.ndarray,
    *,
    attr_dict: dict[str, Any] | None = None,
    mask_dtype: np.dtype | str = np.uint8,
) -> str:
    """Write a masks-only patch bag (features appended by a later extraction pass)."""
    output_path = Path(output_path)
    patch_size = int(masks.shape[-1])
    with MaskOnlyPatchBagWriter(
        output_path,
        patch_size=patch_size,
        mask_dtype=mask_dtype,
        attr_dict=attr_dict,
    ) as writer:
        for idx in range(int(coords.shape[0])):
            writer.append(
                int(coords[idx, 0]),
                int(coords[idx, 1]),
                int(patch_uids[idx]),
                masks[idx],
            )
    print("Successfully saved masks-only patch bag:", output_path)
    return str(output_path)


def append_features_to_patch_bag_h5(
    output_path: str | Path,
    feature_obj: Any,
    *,
    feature_dtype: np.dtype | str = np.float16,
    attr_dict: dict[str, Any] | None = None,
) -> str:
    """Append feature tensors to an existing masks-only patch bag."""
    output_path = Path(output_path)
    if not is_masks_only_patch_bag_h5(output_path):
        raise ValueError(f"{output_path} is not a masks-only patch bag")

    feat_dtype = np.dtype(feature_dtype)
    with h5py.File(output_path, "a") as handle:
        n_patches = int(handle["coords"].shape[0])
        if "features" in handle and len(handle["features"].keys()) > 0:
            raise ValueError(f"{output_path} already contains features")

        feature_group = handle.create_group("features")
        layer_names: list[str] = []
        if isinstance(feature_obj, dict):
            for name, tensor in feature_obj.items():
                arr = _to_numpy_array(tensor).astype(feat_dtype, copy=False)
                if int(arr.shape[0]) != n_patches:
                    raise ValueError(
                        f"feature layer {name} has {arr.shape[0]} rows, expected {n_patches}"
                    )
                feature_group.create_dataset(
                    name,
                    data=arr,
                    chunks=(1,) + arr.shape[1:],
                    dtype=feat_dtype,
                )
                layer_names.append(str(name))
        else:
            arr = _to_numpy_array(feature_obj).astype(feat_dtype, copy=False)
            if int(arr.shape[0]) != n_patches:
                raise ValueError(f"features have {arr.shape[0]} rows, expected {n_patches}")
            feature_group.create_dataset(
                "features",
                data=arr,
                chunks=(1,) + arr.shape[1:],
                dtype=feat_dtype,
            )
            layer_names = ["features"]

        handle.attrs["bag_stage"] = PATCH_BAG_STAGE_COMPLETE
        handle.attrs["feature_layers"] = json.dumps(layer_names)
        handle.attrs["feature_dtype"] = str(feat_dtype)
        if attr_dict:
            for key, value in attr_dict.items():
                handle.attrs[key] = value

    print("Successfully appended features to patch bag:", output_path)
    return str(output_path)


def patch_bag_feature_row_count(path: str | Path) -> int:
    """Number of patch rows that already have feature tensors stored."""
    path = Path(path)
    if not path.is_file():
        return 0
    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                return 0
            n_coords = int(handle["coords"].shape[0])
            if "features" not in handle or not isinstance(handle["features"], h5py.Group):
                return 0
            if len(handle["features"].keys()) == 0:
                return 0
            return min(int(next(iter(handle["features"].values())).shape[0]), n_coords)
    except OSError:
        return 0


def load_patch_bag_extracted_keys(path: str | Path, slide_id: str) -> set[tuple[str, int, int]]:
    """Return (slide_id, x, y) keys whose features are already stored in the bag."""
    path = Path(path)
    keys: set[tuple[str, int, int]] = set()
    if not path.is_file():
        return keys
    try:
        with h5py.File(path, "r") as handle:
            n_rows = patch_bag_feature_row_count(path)
            if n_rows <= 0:
                return keys
            coords = handle["coords"][:n_rows]
            for row in coords:
                keys.add((slide_id, int(row[0]), int(row[1])))
    except OSError:
        return keys
    return keys


def load_patch_bag_unique_rows(path: str | Path, slide_id: str, wsi_path: str = "") -> list[dict[str, Any]]:
    """Load per-row patch metadata from an existing bag for resume."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                return rows
            coords = np.asarray(handle["coords"])
            patch_uids = (
                np.asarray(handle["patch_uids"])
                if "patch_uids" in handle
                else np.arange(coords.shape[0], dtype=np.int64)
            )
            for idx in range(coords.shape[0]):
                rows.append(
                    {
                        "slide_id": slide_id,
                        "x": int(coords[idx, 0]),
                        "y": int(coords[idx, 1]),
                        "wsi_path": wsi_path,
                        "patch_uid": int(patch_uids[idx]),
                    }
                )
    except OSError:
        return rows
    return rows


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    print('Successfully saved:', output_path)
    return output_path