import os
import sys
import io
import random
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def partition(base_dir=None, out_dir=None):
    if base_dir is None:
        if len(sys.argv) > 1:
            base_dir = os.path.abspath(sys.argv[1])
        else:
            base_dir = os.path.abspath('datasets/iiit_indic_hw_uc/tamil/file')
    
    if out_dir is None:
        if len(sys.argv) > 2:
            out_dir = os.path.abspath(sys.argv[2])
        else:
            out_dir = base_dir

    os.makedirs(out_dir, exist_ok=True)
    train_gt_p = os.path.join(base_dir, 'train_gt.txt')
    val_gt_p = os.path.join(base_dir, 'val_gt.txt')
    test_gt_p = os.path.join(base_dir, 'test_gt.txt')

    print("=== Creating Reproducible Document-Grouped 85/15 Split ===")
    print(f"Reading train candidates: {train_gt_p}")
    print(f"Reading val candidates: {val_gt_p}")
    print(f"Quarantining test partition: {test_gt_p} (STRICTLY EXCLUDED)")

    if not os.path.exists(train_gt_p) or not os.path.exists(val_gt_p):
        raise FileNotFoundError(f"Missing train_gt.txt or val_gt.txt in {base_dir}")

    # Read train and val candidates
    all_pairs = []
    with open(train_gt_p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(line)
    train_count = len(all_pairs)

    with open(val_gt_p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(line)
    val_count = len(all_pairs) - train_count

    print(f"Total candidate word samples pooled: {len(all_pairs)} (train={train_count}, val={val_count})")

    # Group by Document ID
    doc_to_lines = defaultdict(list)
    for line in all_pairs:
        rel_img = line.split('\t')[0]
        filename = rel_img.split('/')[-1]
        doc_id = filename.split('_')[0]
        doc_to_lines[doc_id].append(line)

    unique_docs = sorted(doc_to_lines.keys())
    print(f"Total unique document IDs: {len(unique_docs)}")

    # Reproducible seeded shuffle
    rng = random.Random(42)
    shuffled_docs = list(unique_docs)
    rng.shuffle(shuffled_docs)

    # 85% train, 15% val split on document IDs
    total_docs = len(shuffled_docs)
    train_doc_count = int(round(total_docs * 0.85))
    train_docs = set(shuffled_docs[:train_doc_count])
    val_docs = set(shuffled_docs[train_doc_count:])

    print(f"Document split: {len(train_docs)} train documents, {len(val_docs)} val documents")
    assert len(train_docs & val_docs) == 0, "Document overlap detected!"

    # Compile samples
    train_split_lines = []
    val_split_lines = []

    for doc_id, lines in doc_to_lines.items():
        if doc_id in train_docs:
            train_split_lines.extend(lines)
        elif doc_id in val_docs:
            val_split_lines.extend(lines)
        else:
            raise ValueError(f"Document {doc_id} not assigned!")

    print(f"Sample split: {len(train_split_lines)} train samples ({len(train_split_lines)/len(all_pairs)*100:.2f}%), "
          f"{len(val_split_lines)} val samples ({len(val_split_lines)/len(all_pairs)*100:.2f}%)")

    # Check for image overlap
    train_imgs = set(l.split('\t')[0] for l in train_split_lines)
    val_imgs = set(l.split('\t')[0] for l in val_split_lines)
    assert len(train_imgs & val_imgs) == 0, "Image overlap detected!"

    # Check test quarantine isolation if test_gt exists
    if os.path.exists(test_gt_p):
        with open(test_gt_p, 'r', encoding='utf-8') as f:
            test_imgs = set(l.strip().split('\t')[0] for l in f if l.strip())
        assert len(train_imgs & test_imgs) == 0, "CRITICAL: Test image found in train split!"
        assert len(val_imgs & test_imgs) == 0, "CRITICAL: Test image found in val split!"
        print("Test partition quarantine: 100% ISOLATED (Zero test samples in train or val).")

    # Save to disk
    out_train_p = os.path.join(out_dir, 'train_doc_split_85_15.txt')
    out_val_p = os.path.join(out_dir, 'val_doc_split_85_15.txt')

    with open(out_train_p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_split_lines) + '\n')
    with open(out_val_p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_split_lines) + '\n')

    print(f"Saved: {out_train_p} ({len(train_split_lines)} lines)")
    print(f"Saved: {out_val_p} ({len(val_split_lines)} lines)")
    print("Partitioning successfully complete!")

if __name__ == '__main__':
    partition()
