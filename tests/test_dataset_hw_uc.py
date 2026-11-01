"""
test_dataset_hw_uc.py - Unit tests for IIITIndicHWUCDataset and Document Partitioning.
"""

import os
import unittest
import tempfile
import numpy as np
from PIL import Image
import torch

from src.data.vocab import TamilCTCCharacterVocabulary
from src.data.dataset_hw_uc import IIITIndicHWUCDataset
from src.data.dataset import pad_collate_fn

class TestIIITIndicHWUCDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_path = os.path.abspath('checkpoints/crnn/tamil_ctc_char_vocab.json')
        cls.vocab = TamilCTCCharacterVocabulary(vocab_file=cls.vocab_path)
        cls.real_dataset_root = os.path.abspath('datasets/iiit_indic_hw_uc/tamil')

    def test_vocab_initialization(self):
        """Verify vocabulary loads 73 tokens (72 characters + blank)."""
        self.assertEqual(len(self.vocab), 73)
        self.assertEqual(self.vocab.blank_id, 0)
        self.assertIn('அ', self.vocab.char2idx)
        self.assertIn('க', self.vocab.char2idx)
        self.assertIn('்', self.vocab.char2idx)

    def test_synthetic_sample_loading_and_cleaning(self):
        """Verify tab-separated parsing, punctuation cleaning, and OOV handling with synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy images
            img1_path = os.path.join(tmpdir, "img1.jpg")
            img2_path = os.path.join(tmpdir, "img2.jpg")
            img3_path = os.path.join(tmpdir, "img3.jpg")
            Image.new('RGB', (200, 100), color=(255, 255, 255)).save(img1_path)
            Image.new('L', (300, 80), color=(200,)).save(img2_path)
            Image.new('RGB', (150, 90), color=(100, 100, 100)).save(img3_path)

            # Create synthetic GT file
            # img1: clean Tamil word
            # img2: Tamil word with leading and trailing English quotes/period
            # img3: Word with English letters (OOV)
            gt_file = os.path.join(tmpdir, "custom_gt.txt")
            with open(gt_file, 'w', encoding='utf-8') as f:
                f.write(f"img1.jpg\tஅவர்களின்\n")
                f.write(f"img2.jpg\t\"நிலை.\"\n")
                f.write(f"img3.jpg\tTamilText\n")

            ds = IIITIndicHWUCDataset(
                root_dir=tmpdir,
                gt_path=gt_file,
                vocab=self.vocab,
                filter_oov=True,
                clean_punctuation=True
            )

            # img3 should be dropped due to OOV English letters; img1 and img2 should be kept
            self.assertEqual(len(ds), 2)
            self.assertEqual(len(ds.dropped_records), 1)
            self.assertIn("OOV", ds.dropped_records[0][2])

            # Test item 0
            item0 = ds[0]
            self.assertEqual(item0['text'], 'அவர்களின்')
            self.assertEqual(item0['image'].shape[1], 64) # height 64
            self.assertGreater(item0['width'], 16)
            self.assertEqual(len(item0['target']), item0['target_length'])

            # Test item 1 (cleaned punctuation)
            item1 = ds[1]
            self.assertEqual(item1['text'], 'நிலை')

    def test_real_dataset_sample(self):
        """Verify loading 20 samples from real extracted IIIT-Indic-HW-UC train set."""
        if not os.path.exists(self.real_dataset_root):
            self.skipTest(f"Dataset root not found at {self.real_dataset_root}")

        ds = IIITIndicHWUCDataset(
            root_dir=self.real_dataset_root,
            split='train',
            vocab=self.vocab,
            max_samples=20
        )

        self.assertEqual(len(ds), 20)
        item = ds[0]
        self.assertEqual(item['image'].shape[0], 1)
        self.assertEqual(item['image'].shape[1], 64)
        self.assertTrue(0.0 <= item['image'].min().item() <= 1.0)
        self.assertTrue(0.0 <= item['image'].max().item() <= 1.0)

    def test_pad_collate_fn_compatibility(self):
        """Verify variable-width batch collation with IIITIndicHWUCDataset."""
        if not os.path.exists(self.real_dataset_root):
            self.skipTest(f"Dataset root not found at {self.real_dataset_root}")

        ds = IIITIndicHWUCDataset(
            root_dir=self.real_dataset_root,
            split='train',
            vocab=self.vocab,
            max_samples=8
        )

        batch = [ds[i] for i in range(4)]
        collated = pad_collate_fn(batch)

        self.assertEqual(collated['images'].shape[0], 4)
        self.assertEqual(collated['images'].shape[1], 1)
        self.assertEqual(collated['images'].shape[2], 64)
        self.assertEqual(collated['images'].shape[3] % 4, 0) # Multiple of 4
        self.assertEqual(len(collated['input_lengths']), 4)
        self.assertEqual(collated['input_lengths'][0].item(), collated['images'].shape[3] // 4)

    def test_document_split_isolation(self):
        """Verify that the generated 85/15 document split has zero document ID and zero image overlap."""
        train_split_p = os.path.join(self.real_dataset_root, 'file', 'train_doc_split_85_15.txt')
        val_split_p = os.path.join(self.real_dataset_root, 'file', 'val_doc_split_85_15.txt')
        test_gt_p = os.path.join(self.real_dataset_root, 'file', 'test_gt.txt')

        if not (os.path.exists(train_split_p) and os.path.exists(val_split_p)):
            self.skipTest("Document split files not generated yet.")

        def load_entries(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            imgs = set(l.split('\t')[0] for l in lines)
            docs = set(p.split('/')[-1].split('_')[0] for p in imgs)
            return imgs, docs

        train_imgs, train_docs = load_entries(train_split_p)
        val_imgs, val_docs = load_entries(val_split_p)
        test_imgs, _ = load_entries(test_gt_p)

        # Zero overlap assertions
        self.assertEqual(len(train_docs & val_docs), 0, "Document overlap between train and val!")
        self.assertEqual(len(train_imgs & val_imgs), 0, "Image overlap between train and val!")
        self.assertEqual(len(train_imgs & test_imgs), 0, "Quarantined test image in train!")
        self.assertEqual(len(val_imgs & test_imgs), 0, "Quarantined test image in val!")

        self.assertGreaterEqual(len(train_imgs), 140000)
        self.assertGreaterEqual(len(val_imgs), 20000)

if __name__ == '__main__':
    unittest.main()
