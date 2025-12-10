# Cultural Specificity Classifier

A transformer-based NLP system for classifying items by their degree of cultural specificity.

## Problem Statement

Understanding cultural context is essential for building inclusive AI systems. Items, concepts, and entities vary in how culturally specific they are — some are universally recognized while others are meaningful only within particular cultural contexts.

This project addresses the challenge of automatically classifying items into three levels of cultural specificity:

| Category | Definition | Examples |
|----------|------------|----------|
| **Cultural Agnostic** | Universally known items with no specific cultural association | car, bridge, water, mathematics |
| **Cultural Representative** | Items originating from a specific culture but recognized globally | pizza (Italian), sushi (Japanese), yoga (Indian) |
| **Cultural Exclusive** | Items known primarily within a specific culture or region | caponata (Sicilian), xiaolongbao (Shanghai), poutine (Québécois) |

### Challenges

1. **Ambiguous Boundaries**: The distinction between "representative" and "exclusive" is inherently subjective and context-dependent
2. **Class Imbalance**: Uneven distribution across categories in training data
3. **Limited Signal**: Short item descriptions may lack explicit cultural indicators
4. **Temporal Drift**: Cultural specificity evolves as items become globalized over time

### Applications

- Bias detection and mitigation in language models
- Content localization and recommendation systems
- Cross-cultural communication tools
- Cultural heritage preservation and documentation

## Dataset

| Split | Samples | Distribution |
|-------|---------|--------------|
| Train | 6,251 | Exclusive 43%, Agnostic 30%, Representative 27% |
| Test | 300 | Agnostic 39%, Representative 36%, Exclusive 25% |

Each sample contains: item identifier, name, description, type (concept/entity), category, and subcategory across 19 domains including food, literature, music, and visual arts.

## Approaches

### Approach 1: Baseline (Selected)

Fine-tuned multilingual DeBERTa-v3-base with standard cross-entropy loss and class weighting.

| Component | Configuration |
|-----------|---------------|
| Model | `microsoft/mdeberta-v3-base` (86M parameters) |
| Input Format | Structured concatenation of name, description, type, category |
| Max Length | 384 tokens |
| Training | 5-fold stratified CV, 8 epochs |
| Optimization | AdamW, cosine LR schedule, label smoothing (0.1) |
| Class Balancing | Inverse frequency weighting |

### Approach 2: Enhanced with Data Augmentation (Archived)

Attempted to improve performance using a larger model with Wikipedia-enriched descriptions.

| Component | Configuration |
|-----------|---------------|
| Model | `microsoft/deberta-v3-large` (304M parameters) |
| Augmentation | Wikipedia summaries, language availability counts, category tags |
| Loss | Focal loss (γ=2.0) for hard example mining |
| Max Length | 512 tokens |
| Training | 12 epochs |

**Outcome**: Achieved higher validation metrics (~83% F1) but degraded test performance due to distribution mismatch — the model learned to rely on Wikipedia features absent during inference. This approach is archived in `archived_augmentation_experiment/`.

**Key Insight**: Data augmentation must be consistently available at both training and inference time. Larger models do not compensate for train-test distribution shifts.

## Results

### **Test Set Performance (Baseline Model)**

| Metric            | Score      |
| ----------------- | ---------- |
| Number of samples | 300        |
| Accuracy          | **0.7733** |
| Macro F1          | **0.7578** |

---

### **Per-class metrics**

| Class                   | Precision | Recall | F1-score | Support |
| ----------------------- | --------- | ------ | -------- | ------- |
| Cultural Agnostic       | 0.87      | 0.93   | 0.90     | 117     |
| Cultural Exclusive      | 0.65      | 0.71   | 0.68     | 76      |
| Cultural Representative | 0.75      | 0.64   | 0.69     | 107     |

| Metric       | Score | Support |
| ------------ | ----- | ------- |
| Accuracy     | 0.77  | 300     |
| Macro Avg    | 0.76  | 300     |
| Weighted Avg | 0.77  | 300     |

---

### **Cross-Validation Performance (5-Fold)**

| Metric   | Mean ± Std          |
| -------- | ------------------- |
| Accuracy | **0.8047 ± 0.0044** |
| F1 Macro | **0.7992 ± 0.0039** |


### Error Analysis

Primary confusion occurs between Cultural Representative and Cultural Exclusive classes (51% of errors), reflecting the inherent ambiguity in distinguishing items with partial global recognition.

## Project Structure

```
├── train.py                # Training pipeline with K-fold CV
├── inference.py            # Batch prediction generation  
├── evaluate.py             # Metrics and error analysis
├── test.csv                # Test dataset
├── results/                # Predictions, confusion matrices, reports
└── archived_augmentation_experiment/
    ├── README.md           # Documentation of failed approach
    ├── augment_data_async.py
    ├── train_enhanced.py
    └── ...
```

## Usage

```bash
# Train model
python train.py

# Generate predictions
python inference.py

# Evaluate against ground truth
python evaluate.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- scikit-learn 1.3+