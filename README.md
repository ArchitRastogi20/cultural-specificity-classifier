# Cultural Specificity Classification

A transformer-based text classification system for categorizing items by cultural specificity.

## Task

Classify items into three categories:
- **Cultural Agnostic**: Universally known items (e.g., "car", "water")
- **Cultural Representative**: Culture-associated but globally recognized (e.g., "pizza", "sushi")
- **Cultural Exclusive**: Known primarily within a specific culture (e.g., "caponata")

## Dataset

| Split | Samples |
|-------|---------|
| Train | 6,251 |
| Test | 300 |

Features: item name, description, type (concept/entity), category, subcategory

## Approach

Fine-tuned `microsoft/mdeberta-v3-base` using 5-fold stratified cross-validation with class-weighted loss and label smoothing to handle class imbalance.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | mDeBERTa-v3-base (86M params) |
| Max Length | 384 tokens |
| Batch Size | 8 (effective: 16 with grad accumulation) |
| Learning Rate | 1.5e-5 |
| Scheduler | Cosine |
| Epochs | 8 |

## Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | 78.67% |
| F1 Macro | 0.7671 |
| F1 Weighted | 0.7827 |
| Cohen's Kappa | 0.6741 |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Cultural Agnostic | 0.883 | 0.966 | 0.922 |
| Cultural Representative | 0.745 | 0.682 | 0.712 |
| Cultural Exclusive | 0.676 | 0.658 | 0.667 |

### Cross-Validation Summary

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.8047 ± 0.0044 |
| F1 Macro | 0.7992 ± 0.0039 |

## Alternative Approach

An enhanced approach using `deberta-v3-large` (304M params) with Wikipedia data augmentation was explored. Despite achieving higher validation metrics (~83% F1), it underperformed on test data due to train-test distribution mismatch. The augmented descriptions contained Wikipedia metadata not present at inference time. This experiment is archived in `archived_augmentation_experiment/`.

## Project Structure

```
├── train.py              # Training pipeline
├── inference.py          # Prediction generation
├── evaluate.py           # Evaluation metrics
├── test.csv              # Test dataset
├── results/              # Predictions and analysis
└── archived_augmentation_experiment/
```

## Usage

```bash
# Training
python train.py

# Inference
python inference.py

# Evaluation
python evaluate.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- scikit-learn