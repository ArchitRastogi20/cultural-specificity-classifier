# Archived: Wikipedia Augmentation Experiment

## Overview

This directory contains an experimental approach that used Wikipedia API data augmentation with a larger model architecture. The experiment was archived due to train-test distribution mismatch issues.

## Approach Summary

| Component | Configuration |
|-----------|---------------|
| Base Model | `microsoft/deberta-v3-large` (304M parameters) |
| Data Augmentation | Wikipedia API (summaries, language counts, categories) |
| Loss Function | Focal Loss + Label Smoothing |
| Max Sequence Length | 512 tokens |
| Training Epochs | 12 |

## Why Archived

The Wikipedia augmentation strategy enriched training descriptions with additional context (Wikipedia summaries, language availability counts, category tags). While this improved cross-validation metrics (~83% F1), it introduced a distribution mismatch:

- **Training data**: Enriched with Wikipedia metadata
- **Test data**: Original descriptions without enrichment

When the augmented model was evaluated on non-augmented test data, performance degraded due to the model expecting features that were not present at inference time.

## Files

| File | Description |
|------|-------------|
| `augment_data_async.py` | Async Wikipedia API fetcher (50 concurrent requests) |
| `train_enhanced.py` | Training script with Focal Loss implementation |
| `check_augmentation.py` | Utility to verify augmentation statistics |
| `train_augmented.csv` | Wikipedia-enriched training data |
| `test_augmented.csv` | Wikipedia-enriched test data |
| `training_results/` | Model checkpoints and fold results |
| `predictions/` | Inference outputs |

## Lessons Learned

1. Data augmentation must be consistently applied at both training and inference time
2. Larger models do not guarantee better performance when data distributions differ
3. Cross-validation metrics can be misleading if test conditions differ from validation

## HuggingFace Model

The trained model is available as a prviate model on HuggingFace