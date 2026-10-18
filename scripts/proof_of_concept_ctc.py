"""
proof_of_concept_ctc.py - Minimal Diagnostic Pipeline Verification.

Loads 16 real handwritten Tamil word samples directly from the raw dataset,
executes forward pass through TamilCRNN, calculates CTCLoss, runs backward pass,
overfits over 30 diagnostic iterations, and decodes predicted Tamil Unicode text.
"""

import os
import io
import zipfile
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.vocab import TamilCTCCharacterVocabulary
from src.models.crnn import TamilCRNN

def preprocess_word_image(pil_img: Image.Image, target_h: int = 64, target_w: int = 512) -> torch.Tensor:
    # 1. Grayscale
    gray = pil_img.convert('L')
    orig_w, orig_h = gray.size
    
    # 2. Scale preserving aspect ratio
    new_w = max(16, min(target_w, int(orig_w * target_h / orig_h)))
    resized = gray.resize((new_w, target_h), Image.Resampling.BILINEAR)
    
    # 3. Canvas pad (white background 255)
    padded = Image.new('L', (target_w, target_h), color=255)
    padded.paste(resized, (0, 0))
    
    # 4. Invert and normalize to [0.0, 1.0] (ink = 1.0, background = 0.0)
    arr = (255.0 - np.array(padded, dtype=np.float32)) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0) # (1, H, W)
    return tensor

def run_poc():
    print("=== STARTING TAMIL WORD OCR PROOF OF CONCEPT (CTC) ===")
    
    # 1. Load Vocab
    vocab_json = 'reports/tamil_ctc_char_vocab.json'
    vocab = TamilCTCCharacterVocabulary()
    vocab.load(vocab_json)
    print(f"[1/6] Loaded CTC vocabulary with {len(vocab)} tokens.")
    
    # 2. Load 16 sample word images from val.zip inside raw archive
    zip_path = r'datasets/raw/tamil_word_dataset.zip'
    with zipfile.ZipFile(zip_path, 'r') as main_z:
        with main_z.open('vocab.txt') as f:
            word_lexicon = f.read().decode('utf-8').splitlines()
            
        with main_z.open('val.txt') as f:
            val_lines = [l.strip() for l in f.read().decode('utf-8').splitlines() if l.strip()]
            
        val_bytes = io.BytesIO(main_z.read('val.zip'))
        with zipfile.ZipFile(val_bytes, 'r') as val_z:
            samples = []
            for i in range(16):
                line = val_lines[i]
                img_path, v_id = line.split(',')
                img_path = img_path.strip()
                v_id = int(v_id.strip())
                word_text = word_lexicon[v_id]
                
                with val_z.open(img_path) as img_file:
                    pil_img = Image.open(img_file)
                    t_img = preprocess_word_image(pil_img, target_h=64, target_w=512)
                    samples.append((t_img, word_text))

    print(f"[2/6] Successfully loaded and preprocessed {len(samples)} real handwritten images.")
    
    # 3. Construct batch tensors
    batch_x = torch.stack([s[0] for s in samples]) # (B, 1, 64, 512)
    B, _, H, W = batch_x.shape
    T = W // 4 # 512 // 4 = 128
    
    targets = []
    target_lengths = []
    for s in samples:
        enc = vocab.encode(s[1])
        targets.extend(enc)
        target_lengths.append(len(enc))
        
    targets_t = torch.tensor(targets, dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths_t = torch.full(size=(B,), fill_value=T, dtype=torch.long)
    
    print(f"[3/6] Batch shape: {batch_x.shape}, Time-steps T: {T}, Targets total: {len(targets)}")
    
    # 4. Instantiate Model & Loss
    model = TamilCRNN(num_classes=len(vocab), in_channels=1, hidden_size=256)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Overfit Diagnostic Training Loop (30 iterations)
    print("[4/6] Executing diagnostic overfitting loop (30 iterations)...")
    for epoch in range(1, 31):
        model.train()
        optimizer.zero_grad()
        
        logits = model(batch_x) # (T, B, C)
        log_probs = logits.log_softmax(2)
        
        loss = criterion(log_probs, targets_t, input_lengths_t, target_lengths_t)
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN / Inf loss detected at epoch {epoch}!")
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Iteration {epoch:2d}/30 | CTC Loss: {loss.item():.4f}")
            
    print(f"[5/6] Overfitting completed. Final CTC loss: {loss.item():.4f}")
    
    # 6. Test CTC Greedy Decoding
    model.eval()
    with torch.no_grad():
        logits = model(batch_x) # (T, B, C)
        preds = logits.argmax(dim=2) # (T, B)
        preds = preds.permute(1, 0).cpu().numpy() # (B, T)
        
    print("[6/6] Diagnostic prediction samples:")
    for i in range(3):
        pred_indices = preds[i].tolist()
        decoded_text = vocab.ctc_decode(pred_indices)
        gt_text = samples[i][1]
        print(f"   Sample {i+1}:")
        print(f"      Ground Truth: {gt_text.encode('unicode_escape').decode('latin1')} (len {len(gt_text)})")
        print(f"      Prediction:   {decoded_text.encode('unicode_escape').decode('latin1')} (len {len(decoded_text)})")
        
    # 7. Save verification checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/poc_crnn_overfit.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(vocab),
        'hidden_size': 256,
        'final_loss': loss.item()
    }, ckpt_path)
    print(f"[OK] Saved POC checkpoint to {ckpt_path}")
    print("=== TAMIL WORD OCR PROOF OF CONCEPT PASSED SUCCESSFULLY ===")

if __name__ == '__main__':
    run_poc()
