"""
test_dataset.py - Unit tests for uTHCD dataset loader and HDF5 integrity.
"""

import os
import unittest
import numpy as np
import torch
import h5py

from model.dataset import uTHCDDataset

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hdf5_path = 'data/hdf5_uTHCD_compressed.h5'
        if not os.path.exists(cls.hdf5_path):
            raise unittest.SkipTest("HDF5 dataset file not found.")

    def test_hdf5_structure_and_keys(self):
        """Verify HDF5 groups and dataset dimensions."""
        with h5py.File(self.hdf5_path, 'r') as f:
            self.assertIn('Train Data', f)
            self.assertIn('Test Data', f)

            g_train = f['Train Data']
            g_test = f['Test Data']

            self.assertIn('x_train', g_train)
            self.assertIn('y_train', g_train)
            self.assertIn('x_test', g_test)
            self.assertIn('y_test', g_test)

            self.assertEqual(g_train['x_train'].shape, (62870, 64, 64))
            self.assertEqual(g_train['y_train'].shape, (62870,))
            self.assertEqual(g_test['x_test'].shape, (28080, 64, 64))
            self.assertEqual(g_test['y_test'].shape, (28080,))

    def test_dataset_splits_and_lengths(self):
        """Verify deterministic train, val, and test splits."""
        ds_train = uTHCDDataset(self.hdf5_path, split='train')
        ds_val = uTHCDDataset(self.hdf5_path, split='val')
        ds_test = uTHCDDataset(self.hdf5_path, split='test')

        self.assertEqual(len(ds_train), 55000)
        self.assertEqual(len(ds_val), 7870)
        self.assertEqual(len(ds_test), 28080)
        self.assertEqual(len(ds_train) + len(ds_val), 62870)

    def test_dataset_item_shape_and_range(self):
        """Verify tensor item shape, type, and normalized pixel values."""
        ds_test = uTHCDDataset(self.hdf5_path, split='test')
        img_tensor, label = ds_test[0]

        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (1, 64, 64))
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertGreaterEqual(img_tensor.min().item(), 0.0)
        self.assertLessEqual(img_tensor.max().item(), 1.0)
        self.assertGreaterEqual(label, 0)
        self.assertLessEqual(label, 155)

    def test_dataset_subset_max_samples(self):
        """Verify subset loading for resource-constrained development."""
        ds_sub = uTHCDDataset(self.hdf5_path, split='train', max_samples=50)
        self.assertEqual(len(ds_sub), 50)
        img_tensor, label = ds_sub[49]
        self.assertEqual(img_tensor.shape, (1, 64, 64))

if __name__ == '__main__':
    unittest.main()
