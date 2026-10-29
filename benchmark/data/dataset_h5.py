from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import h5py
import os
import glob
def _resolve_native_mpp(wsi, fallback_mpp):
    # irreducible cflow, using cdg fallback
    """Read native microns-per-pixel from a WSI handle, or use fallback."""
    if wsi is None:
        return float(fallback_mpp)
    try:
        mpp = getattr(wsi, 'mpp', None)
        if mpp is not None:
            if isinstance(mpp, (tuple, list, np.ndarray)):
                return float((float(mpp[0]) + float(mpp[1])) / 2.0)
            else:
                return float(mpp)
    except (TypeError, ValueError, AttributeError):
        pass
    props = getattr(wsi, 'properties', None)
    if props is not None:
        try:
            if 'openslide.mpp-x' in props:
                mpp_x = float(props['openslide.mpp-x'])
                mpp_y = float(props['openslide.mpp-y'])
                return (mpp_x + mpp_y) / 2.0
        except (KeyError, TypeError, ValueError):
            pass
    return float(fallback_mpp)
def _mpp_scale(native_mpp, target_mpp):
    if target_mpp is None or target_mpp <= 0:
        return 1.0
    else:
        return float(target_mpp) / float(native_mpp)
COLOR_CORRECTION_FLAG = False
if 'COLOR_CORRECTION_FLAG' in os.environ:
    if os.environ['COLOR_CORRECTION_FLAG'].lower() in ['1', 'true', 'yes']:
        COLOR_CORRECTION_FLAG = True
def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return trnsfrms_val
class Whole_Slide_Bag(Dataset):
    pass
    def __init__(self, file_path, pretrained=False, custom_transforms=None, target_patch_size=(-1)):
        """\n        Args:\n            file_path (string): Path to the .h5 file containing patched data.\n            pretrained (bool): Use ImageNet transforms\n            custom_transforms (callable, optional): Optional transform to be applied on a sample\n        """
        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            dset = f['imgs']
            self.length = len(dset)
        self.summary()
    def __len__(self):
        return self.length
    def summary(self):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            dset = hdf5_file['imgs']
            for name, value in dset.attrs.items():
                print(name, value)
        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]
        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return (img, coord)
class Whole_Slide_Bag_FP(Dataset):
    pass
    def __init__(self, file_path, wsi, pretrained=False, custom_transforms=None, custom_downsample=1, target_patch_size=(-1), fast_read=False, target_mpp=None, fallback_mpp=0.25, is_patch_dataset=False, patch_image_dir=None, mask_path=None):
        """\n        Args:\n            file_path (string): Path to the .h5 file containing patched data.\n            pretrained (bool): Use ImageNet transforms\n            custom_transforms (callable, optional): Optional transform to be applied on a sample\n            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)\n            target_patch_size (int): Custom defined image size before embedding\n            target_mpp (float, optional): Target MPP for the foundation model (e.g. 0.25, 0.50).\n            fallback_mpp (float): Assumed native MPP when metadata or patch mode default applies.\n            is_patch_dataset (bool): If True, load static patches (PIL) instead of WSI read_region.\n            patch_image_dir (str, optional): Root directory of pre-cut patch images.\n        """
        self.pretrained = pretrained
        self.wsi = wsi
        self.is_patch_dataset = bool(is_patch_dataset)
        self.target_mpp = target_mpp
        self.patch_image_dir = patch_image_dir
        self.mask_path = mask_path
        if COLOR_CORRECTION_FLAG:
            if hasattr(self.wsi, 'apply_color_correction'):
                try:
                    print('Using color correction for WSI:', file_path)
                    self.wsi.apply_color_correction()
                except Exception as e:
                    print('Failed to apply color correction for WSI:', file_path)
                    print('Error message:', str(e))
            else:
                print('Color correction flag is set but WSI has no color correction method:', file_path)
                print('The reason could be that the WSI is not in a supported format for color correction.')
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms
        self.file_path = file_path
        self.fast_read = fast_read
        if fast_read:
            print('Loading coord file:', self.file_path)
            with h5py.File(self.file_path, 'r') as hdf5_file:
                self.coords = hdf5_file['coords'][:]
        with h5py.File(self.file_path, 'r') as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = int(f['coords'].attrs['patch_size'])
            self.length = len(dset)
            self._has_stored_imgs = 'imgs' in f
            self._has_stored_masks = 'masks' in f
            self._full_mask = f['mask_full'][:] if 'mask_full' in f else None
            self._mask_attrs = dict(f['masks'].attrs) if 'masks' in f else {}
        self._external_mask = self._load_external_mask(mask_path)
        if self.fast_read and (self._full_mask is not None or self._external_mask is not None):
                mask_shape = (self._full_mask if self._full_mask is not None else self._external_mask).shape
                print('Loaded full-slide mask for cropping:', mask_shape)
        if self.is_patch_dataset:
            self.native_mpp = float(fallback_mpp)
        else:
            self.native_mpp = _resolve_native_mpp(self.wsi, fallback_mpp)
        self.mpp_scale = _mpp_scale(self.native_mpp, self.target_mpp)
        if target_patch_size > 0:
            self.output_patch_size = int(target_patch_size)
        else:
            self.output_patch_size = self.patch_size
        if target_mpp is None or target_mpp <= 0:
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, target_patch_size)
            else:
                if custom_downsample > 1:
                    self.target_patch_size = (self.patch_size // custom_downsample,) * 2
                else:
                    self.target_patch_size = None
        else:
            self.target_patch_size = None
        if self.is_patch_dataset and (not self._has_stored_imgs) and (not self.patch_image_dir):
            raise ValueError('is_patch_dataset=True requires coords HDF5 with an \"imgs\" dataset or a valid patch_image_dir.')
        else:
            self.summary()
    def __len__(self):
        return self.length
    def summary(self):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            dset = hdf5_file['coords']
            for name, value in dset.attrs.items():
                print(name, value)
        print('\nfeature extraction settings')
        print('native_mpp:', self.native_mpp)
        print('target_mpp:', self.target_mpp)
        print('mpp_scale:', self.mpp_scale)
        print('output_patch_size:', self.output_patch_size)
        print('is_patch_dataset:', self.is_patch_dataset)
        print('has_stored_masks:', getattr(self, '_has_stored_masks', False))
        print('has_full_mask:', self._full_mask is not None or self._external_mask is not None)
        print('target patch size (legacy):', self.target_patch_size)
        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
    def _load_external_mask(self, mask_path):
        """Load an optional slide-level semantic mask aligned to level-0 WSI coords."""
        if mask_path is None:
            return
        else:
            mask_path = str(mask_path)
            if not os.path.isfile(mask_path):
                return
            else:
                with Image.open(mask_path) as handle:
                    return np.asarray(handle.convert('L'))
    def _crop_mask_array(self, mask_array, coord, read_size):
        x, y = (int(coord[0]), int(coord[1]))
        read_size = int(read_size)
        height, width = mask_array.shape[:2]
        src_left = max(x, 0)
        src_top = max(y, 0)
        src_right = min(x + read_size, width)
        src_bottom = min(y + read_size, height)
        if src_right <= src_left or src_bottom <= src_top:
            return np.zeros((read_size, read_size), dtype=mask_array.dtype)
        else:
            out = np.zeros((read_size, read_size), dtype=mask_array.dtype)
            dst_left = src_left - x
            dst_top = src_top - y
            patch = mask_array[src_top:src_bottom, src_left:src_right]
            out[dst_top:dst_top + patch.shape[0], dst_left:dst_left + patch.shape[1]] = patch
            return out
    def _load_patch_image_pil(self, idx, coord):
        """Load a pre-cut patch via PIL from HDF5 imgs or patch_image_dir."""
        if self._has_stored_imgs:
            with h5py.File(self.file_path, 'r') as hdf5_file:
                patch_arr = hdf5_file['imgs'][idx]
            return Image.fromarray(patch_arr).convert('RGB')
        else:
            x, y = (int(coord[0]), int(coord[1]))
            patterns = [os.path.join(self.patch_image_dir, f'{x}_{y}.*'), os.path.join(self.patch_image_dir, f'{x}x{y}.*'), os.path.join(self.patch_image_dir, f'{x}-{y}.*'), os.path.join(self.patch_image_dir, str(idx) + '.*')]
            for pattern in patterns:
                matches = sorted(glob.glob(pattern))
                if matches:
                    return Image.open(matches[0]).convert('RGB')
            raise FileNotFoundError(f'No patch image found for coord ({x}, {y}) under {self.patch_image_dir}')
    def _load_patch_mask(self, idx, coord, read_size):
        """Load mask aligned with the image read window (stored patch or full-slide crop)."""
        if self._has_stored_masks:
            with h5py.File(self.file_path, 'r') as hdf5_file:
                return np.asarray(hdf5_file['masks'][idx])
        mask_array = self._full_mask if self._full_mask is not None else self._external_mask
        if mask_array is not None:
            return self._crop_mask_array(mask_array, coord, read_size)
    def _apply_geometry(self, img, mask, read_size):
        """Pad if needed and resize to fixed output_patch_size at target MPP."""
        from utils.resize_ops import apply_mpp_patch_geometry
        if self.target_mpp is not None and self.target_mpp > 0:
            return apply_mpp_patch_geometry(img, mask, read_size, self.output_patch_size)
        else:
            out_size = self.output_patch_size
            if self.target_patch_size is not None:
                out_size = self.target_patch_size[0]
            if out_size and (img.size[0] != out_size or img.size[1] != out_size):
                from utils.resize_ops import resize_image_mask_pair
                return resize_image_mask_pair(img, mask, (out_size, out_size))
            else:
                return (img, mask)
    @property
    def has_masks(self):
        return self._has_stored_masks or self._full_mask is not None or self._external_mask is not None
    def __getitem__(self, idx):
        if self.fast_read:
            coord = self.coords[idx]
        else:
            with h5py.File(self.file_path, 'r') as hdf5_file:
                coord = hdf5_file['coords'][idx]
        if self.target_mpp is not None and self.target_mpp > 0:
            read_size = int(round(self.output_patch_size * float(self.target_mpp) / float(self.native_mpp)))
        else:
            read_size = int(round(self.patch_size * self.mpp_scale))
        read_size = max(read_size, 1)
        mask = self._load_patch_mask(idx, coord, read_size)
        if self.is_patch_dataset:
            img = self._load_patch_image_pil(idx, coord)
        else:
            try:
                img = self.wsi.read_region(tuple(coord), self.patch_level, (read_size, read_size)).convert('RGB')
            except Exception as exc:
                raise RuntimeError('Failed to read WSI region at {},{} level {} from {}'.format(int(coord[0]), int(coord[1]), self.patch_level, self.file_path)) from exc
        img, mask = self._apply_geometry(img, mask, read_size)
        img = self.roi_transforms(img)
        if mask is not None:
            return (img, coord, mask)
        else:
            return (img, coord, None)
class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path):
        import pandas as pd
        self.df = pd.read_csv(csv_path, dtype={'case_id': str, 'slide_id': str})
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
