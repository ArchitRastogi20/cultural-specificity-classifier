# Non-LM Cultural Classification Results

## Best Model: Tuned XGBoost

**Model file**: `models/cultural_classifier_tuned.pkl`

### Test Performance (valid.csv, 300 samples)

| Metric | Score |
|--------|-------|
| Accuracy | 72.67% |
| F1 Macro | 0.7066 |
| F1 Weighted | 0.7118 |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Cultural Agnostic | 0.757 | 0.932 | 0.835 |
| Cultural Exclusive | 0.625 | 0.790 | 0.698 |
| Cultural Representative | 0.817 | 0.458 | 0.587 |

### Feature Importance (Top 5)

1. num_geographic_properties (21.67)
2. has_country (18.05)
3. cultural_x_geographic (10.28)
4. geographic_ratio (9.01)
5. has_origin_country (6.67)

### Model Comparison

| Model | Validation F1 | Test F1 |
|-------|---------------|---------|
| XGBoost (default) | 0.6761 | - |
| **XGBoost (tuned)** | **0.7004** | **0.7066** |
| Ensemble (4 models) | 0.6891 | - |

### Wikipedia Feature Impact

| Approach | F1 Score | Improvement |
|----------|----------|-------------|
| Baseline (text only) | 0.5063 | - |
| With Wikipedia features | 0.7066 | **+39.6%** |
