"""
dataset.py - Efficient, lazy-loading PyTorch Dataset for uTHCD HDF5 data.

Features:
- Streaming access directly from HDF5 file (minimal RAM usage, suitable for 4 GB RAM PC).
- Safe multi-worker / single-worker handle opening.
- Paper-grounded deterministic train/val/test splits:
  * train: first 55,000 samples of 'Train Data' (x_train[:-7870])
  * val:   last 7,870 samples of 'Train Data' (x_train[-7870:])
  * test:  all 28,080 samples of 'Test Data' (x_test)
- Configurable `max_samples` for smoke testing and low-resource experimentation.
"""

import os
from typing import Optional, Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model.preprocessing import preprocess_raw_dataset_image

class uTHCDDataset(Dataset):
    """
    PyTorch Dataset accessing uTHCD HDF5 file lazily.
    """
    def __init__(self,
                 hdf5_path: str = 'data/hdf5_uTHCD_compressed.h5',
                 split: str = 'train',
                 max_samples: Optional[int] = None,
                 transform=None,
                 seed: int = 42):
        """
        Args:
            hdf5_path: Path to the HDF5 file.
            split: One of 'train', 'val', or 'test'.
            max_samples: Optional limit on the number of samples (for fast smoke test / small experiments).
            transform: Optional torchvision or custom callable transform.
            seed: Random seed for deterministic sample sub-selection if max_samples is given.
        """
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 dataset not found at '{hdf5_path}'")

        self.hdf5_path = hdf5_path
        self.split = split.lower()
        self.transform = transform
        self.seed = seed
        self._h5_file = None

        if self.split not in ('train', 'val', 'test'):
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        # Inspect lengths and dataset keys flexibly
        with h5py.File(self.hdf5_path, 'r') as f:
            tr_grp = f['Train Data']
            te_grp = f['Test Data']

            # Detect key names: ('x_train' or 'Images' or 'images')
            tr_x_key = next((k for k in ['x_train', 'Images', 'images'] if k in tr_grp), None)
            tr_y_key = next((k for k in ['y_train', 'Labels', 'labels'] if k in tr_grp), None)
            te_x_key = next((k for k in ['x_test', 'Images', 'images'] if k in te_grp), None)
            te_y_key = next((k for k in ['y_test', 'Labels', 'labels'] if k in te_grp), None)

            if not tr_x_key or not tr_y_key:
                raise KeyError(f"Could not find training images/labels in 'Train Data' of {self.hdf5_path}. Keys: {list(tr_grp.keys())}")
            if not te_x_key or not te_y_key:
                raise KeyError(f"Could not find test images/labels in 'Test Data' of {self.hdf5_path}. Keys: {list(te_grp.keys())}")

            n_train_full = tr_grp[tr_x_key].shape[0]
            n_test = te_grp[te_x_key].shape[0]

        # For 70/30 (62,870 train), exactly 7,870 is the paper validation partition
        # For other splits (80/20, 85/15, 90/10), reserve ~10% of train for validation (balanced across classes)
        if n_train_full == 62870:
            val_size = 7870
        else:
            # Default ~10% of training set for validation
            val_size = int(round(n_train_full * 0.10))

        train_size = n_train_full - val_size

        if self.split == 'train':
            self.group_name = 'Train Data'
            self.start_idx = 0
            self.total_len = train_size
        elif self.split == 'val':
            self.group_name = 'Train Data'
            self.start_idx = train_size
            self.total_len = val_size
        else:  # test
            self.group_name = 'Test Data'
            self.start_idx = 0
            self.total_len = n_test

        self.tr_x_key = tr_x_key
        self.tr_y_key = tr_y_key
        self.te_x_key = te_x_key
        self.te_y_key = te_y_key

        # Setup indices
        if max_samples and 0 < max_samples < self.total_len:
            rng = np.random.RandomState(self.seed)
            # Sample evenly / reproducibly
            selected_offsets = rng.choice(self.total_len, size=max_samples, replace=False)
            selected_offsets.sort()
            self.indices = self.start_idx + selected_offsets
        else:
            self.indices = np.arange(self.start_idx, self.start_idx + self.total_len)

        self.length = len(self.indices)

    def _open_file(self):
        """Lazy initialization of HDF5 handle per process."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, 'r')
            if self.split in ('train', 'val'):
                self._x = self._h5_file[f'Train Data/{self.tr_x_key}']
                self._y = self._h5_file[f'Train Data/{self.tr_y_key}']
            else:
                self._x = self._h5_file[f'Test Data/{self.te_x_key}']
                self._y = self._h5_file[f'Test Data/{self.te_y_key}']

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        self._open_file()
        real_idx = int(self.indices[idx])

        img_raw = self._x[real_idx]  # shape: (64, 64), uint8
        label = int(self._y[real_idx])

        # Preprocess: normalize to [0, 1] float32 where ink=1.0, background=0.0
        norm_img = preprocess_raw_dataset_image(img_raw)  # (64, 64) float32
        tensor_img = torch.from_numpy(norm_img).unsqueeze(0)  # (1, 64, 64)

        if self.transform:
            tensor_img = self.transform(tensor_img)

        return tensor_img, label

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()
