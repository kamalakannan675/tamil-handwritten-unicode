# Ensemble Implementation & Architecture Guide

## 1. Overview
The **Multi-Model Ensemble Engine** (`app/ensemble.py`) combines predictions from four independently trained Convolutional Neural Network (`uTHCDNet`) checkpoints:
1. **70/30 Split**: Baseline production model (`model/checkpoints/best_model.pt`)
2. **80/20 Split**: Trained on 64,584 samples (`outputs/experiments/80_20/best_model.pt`)
3. **85/15 Split**: Trained on 69,638 samples (`outputs/experiments/85_15/best_model.pt`)
4. **90/10 Split**: Trained on 73,710 samples (`outputs/experiments/90_10/best_model.pt`)

---

## 2. Supported Strategies

The inference engine supports three runtime configurations via the `OCR_MODEL_MODE` environment variable:

### Strategy A: `OCR_MODEL_MODE=single` (Default Production Baseline)
- **Active Model**: 70/30 baseline model only (`best_model.pt`).
- **RAM Footprint**: ~154 MB.
- **Inference Latency**: ~2.3 ms / sample on CPU.
- **Best Use Case**: Memory-constrained cloud environments (e.g. Render 512 MB free tier).

### Strategy B: `OCR_MODEL_MODE=ensemble_equal` (Equal-Weight Probability Averaging)
- **Formula**:
  $$P_{\text{ensemble}} = \frac{1}{4} \left( P_{70} + P_{80} + P_{85} + P_{90} \right)$$
- **RAM Footprint**: ~616 MB (4 models in RAM).
- **Inference Latency**: ~9.2 ms / sample on CPU.

### Strategy C: `OCR_MODEL_MODE=ensemble_weighted` (Validation-Tuned Weighted Fusion)
- **Formula**:
  $$P_{\text{ensemble}} = 0.20 P_{70} + 0.25 P_{80} + 0.25 P_{85} + 0.30 P_{90}$$
- **Weights Rationale**: Derived from validation performance and sample exposure, placing higher confidence on models trained on more extensive representative variations.

---

## 3. High-Availability Fallback Mechanism
If any non-baseline checkpoint is missing, corrupted, or fails during loading:
1. The engine detects the failure without crashing.
2. An informative warning is logged (`[EnsembleInferenceEngine] Warning: Only X/4 models loaded. Activating fallback...`).
3. The engine automatically switches to `single` mode using the verified `best_model.pt`.
4. Health telemetry (`/api/health`) reports `fallback_active: true`.
