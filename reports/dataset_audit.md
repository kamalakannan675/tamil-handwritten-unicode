# Dataset Audit Report: IIIT-INDIC-HW Tamil Word Dataset
**Project:** Tamil Handwritten Word Recognition to Full-Page OCR  
**Date:** 2026-10-18  
**Dataset Archive:** `datasets/raw/tamil_word_dataset.zip` (5,121,780,332 bytes / 4.77 GB)

---

## 1. Executive Summary & Verified Archive Structure

The dataset archive `tamil_word_dataset.zip` contains **103,518 high-resolution offline handwritten Tamil word images** partitioned into official train, validation, and test splits:

| Archive Entry | Compressed Size | Uncompressed Size | Description / Role |
|---|:---:|:---:|---|
| **`train.txt`** | 384,315 B | 1,667,489 B | 75,736 training image-to-word index mappings (`train/X.jpg, vocab_id`) |
| **`train.zip`** | 3.75 GB | 3.75 GB | 75,736 JPEG handwritten word images |
| **`val.txt`** | 55,862 B | 222,874 B | 11,598 validation image-to-word index mappings (`val/X.jpg, vocab_id`) |
| **`val.zip`** | 572.5 MB | 572.5 MB | 11,598 JPEG handwritten word images |
| **`test.txt`** | 79,417 B | 331,578 B | 16,184 test image-to-word index mappings (`test/X.jpg, vocab_id`) |
| **`test.zip`** | 802.3 MB | 802.3 MB | 16,184 JPEG handwritten word images |
| **`vocab.txt`** | 66,650 B | 403,618 B | Lexicon of 13,294 distinct Tamil words |

**Total Dataset Size:** 103,518 word instances (73.2% Train, 11.2% Val, 15.6% Test).

---

## 2. Text Encoding & Vocabulary Verification

1. **Encoding Format:**
   `vocab.txt`, `train.txt`, `val.txt`, and `test.txt` are clean UTF-8 text files without BOM.
2. **Label Structure:**
   Each line in `train.txt`, `val.txt`, and `test.txt` follows the exact format:
   `<image_path>, <vocab_index>` (e.g. `train/1.jpg, 7424`).
   The zero-indexed line number in `vocab.txt` directly provides the authentic Tamil Unicode string for that word (e.g. index 7424 $\to$ `தொடர்ந்தபடிதான்`).
3. **Character Inventory (CTC Tokens):**
   Across all 13,294 words in `vocab.txt`, there are exactly **72 unique Unicode characters**:
   - **71 Tamil Unicode Characters** spanning the Unicode range `0x0B80` to `0x0BFF` (independent vowels, consonants, combining vowel signs, *virama* / *pulli*, and *Ayudha Ezhuthu* `ஃ`).
   - **1 ASCII Punctuation Mark**: Hyphen `-` (`U+002D`).
   - Reserving index `0` for the CTC `<blank>` token yields a compact vocabulary dimension of **73 classes** ($K=73$).
4. **Out-of-Vocabulary (OOV) Analysis:**
   - 11,565 unique words in training.
   - 6,052 unique words in validation (532 words are unseen in training).
   - 8,140 unique words in test (1,198 words are unseen in training).
   - **Critical Architectural Takeaway:** Because over 1,000 test words are unseen combinations of characters, a word-level classification head cannot generalize. A **sequence-to-sequence character-level CRNN (CNN + BiLSTM + CTC)** is essential so the model recognizes individual grapheme transitions and correctly transcribes OOV words.

---

## 3. Image Geometry & Physical Characteristics

Sampled across the dataset, word crops exhibit the following properties:
- **Format:** RGB JPEG (`.jpg`).
- **Aspect Ratio:** Distinctly horizontal word crops, with width-to-height ratios ranging from $1.4:1$ to $11.7:1$ (median aspect ratio: **5.44**).
- **Physical Dimensions:**
  - **Heights:** Uniformly around **242px to 315px** (median: 294px).
  - **Widths:** Ranging from **416px to 2,930px** (median: 1,543px).
- **Backgrounds:** Scanned ruled and unruled paper with realistic pen ink stroke variation.

### Optimal Preprocessing Pipeline for CRNN:
1. Convert RGB to grayscale (single channel).
2. Resize height to a standardized $H=64$ pixels.
3. Scale width proportionally: $W' = \text{round}(W \times 64 / H)$, capped at $W_{\text{max}} = 512$ pixels.
4. Right-pad with background pixels to $W_{\text{max}} = 512$ for efficient GPU mini-batching.
5. Invert and normalize pixel intensities to $[0.0, 1.0]$.

---

## 4. Evaluation of Proposed Architecture (CNN + BiLSTM + CTC)

### Suitability Verdict: **100% SUITABLE**
The dataset is an offline handwritten word transcription dataset. Sequence modeling with Connectionist Temporal Classification (CTC) is the state-of-the-art methodology for this exact format because:
1. CTC eliminates the need for expensive character-level horizontal bounding-box annotations.
2. The 72-token Tamil character inventory is compact, avoiding gradient sparsity.
3. The convolutional backbone downsamples width by a factor of 4 to 8, allowing the BiLSTM to capture contextual pen stroke transitions before CTC decoding.

---

## 5. Scope Distinction: Word Recognition vs. Full-Page OCR

- **Available Dataset:** The IIIT-INDIC-HW dataset provides **isolated word crops**, not full multi-line pages.
- **Full-Page Pipeline Architecture:**
  To achieve complete document / page-level OCR, the deployed system will comprise two decoupled stages:
  $$\text{Input Page Image} \xrightarrow{\text{Stage 1: Layout Engine}} \text{Segmented Lines & Words} \xrightarrow{\text{Stage 2: CRNN CTC}} \text{Tamil Unicode Text}$$
  - **Stage 1 (Layout & Segmentation):** Adaptive thresholding, horizontal projection profiles for line segmentation, and morphological word-space detection.
  - **Stage 2 (Recognition):** The trained CRNN CTC model recognizing each detected word crop.
  - **Stage 3 (Reconstruction):** Line and paragraph assembly in correct top-to-bottom, left-to-right reading order.
