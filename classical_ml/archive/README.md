# Archived Experiments

This directory contains experiments that were conducted during model development but did not make it to the final solution. They are preserved for reference and reproducibility.

---

## Directory Structure

```
archive/
├── models/
│   ├── cultural_classifier.pkl          # Baseline XGBoost
│   └── cultural_classifier_ensemble.pkl # Ensemble approach
└── ensemble_experiment/
    ├── train_ensemble.py                 # Ensemble training script
    └── ensemble_training.log             # Training logs
```

---

## Archived Experiments

### 1. Baseline XGBoost (`cultural_classifier.pkl`)

**Approach:** XGBoost classifier with default hyperparameters trained on Wikipedia/Wikidata features.

**Configuration:**
- Model: XGBoost with GPU acceleration
- Features: 50+ engineered features from Wikipedia API
- No hyperparameter tuning

**Results:**
| Metric | Score |
|--------|-------|
| Validation F1 (Macro) | 0.6761 |
| Validation Accuracy | ~70% |

**Why Archived:** Superseded by hyperparameter-tuned version which achieved +4% F1 improvement.

---

### 2. Ensemble Model (`cultural_classifier_ensemble.pkl`)

**Approach:** Voting ensemble combining four different classifiers to leverage diverse learning strategies.

**Architecture:**
```
Ensemble (Soft Voting)
├── XGBoost (gradient boosting)
├── LightGBM (gradient boosting, leaf-wise)
├── CatBoost (gradient boosting, categorical handling)
└── RandomForest (bagging)
```

**Configuration:**
- Voting: Soft (probability averaging)
- Equal weights for all models
- Each model trained on same Wikipedia features

**Results:**
| Metric | Score |
|--------|-------|
| Validation F1 (Macro) | 0.6891 |
| Validation Accuracy | ~71% |

**Why Archived:** Despite combining four models, the ensemble underperformed the single tuned XGBoost model (0.6891 vs 0.7004 F1). The hyperparameter-tuned XGBoost was both simpler and more effective.

**Key Insight:** For this dataset, careful hyperparameter tuning of a single model outperformed ensemble diversity. The Wikipedia features were strong enough that model architecture mattered less than optimization.

---

## Performance Comparison

| Model | Validation F1 | Complexity | Status |
|-------|---------------|------------|--------|
| Baseline XGBoost | 0.6761 | Low | Archived |
| Ensemble (4 models) | 0.6891 | High | Archived |
| **Tuned XGBoost** | **0.7004** | Medium | **Selected** |

---

## Lessons Learned

1. **Hyperparameter tuning > Ensemble complexity**: A well-tuned single model beat a naive ensemble of four models.

2. **Feature quality matters most**: The Wikipedia/Wikidata features were the primary driver of performance. All models benefited equally from these features.

3. **Diminishing returns from ensembling**: When base models are similar (all tree-based) and features are strong, ensembling provides marginal gains that don't justify the complexity.

4. **Optuna efficiency**: 50 trials of Bayesian optimization found better hyperparameters than grid search would have in reasonable time.

---

## Reproducing Archived Experiments

### Baseline Model
```bash
python train_classifier.py --no-tune
```

### Ensemble Model
```bash
python archive/ensemble_experiment/train_ensemble.py
```

---

## Notes

- All models use the same feature set extracted via `extract_features.py`
- Training data: 6,251 samples with Wikipedia enrichment
- Validation data: 300 samples
- GPU acceleration used where available (XGBoost, CatBoost)

