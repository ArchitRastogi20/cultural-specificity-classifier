# Cultural Specificity Classifier (Classical ML)

A non–language-model-based system for classifying items by their degree of cultural specificity using engineered features and classical machine learning.

---

## Problem Statement

Items vary in how culturally specific they are. Some are universally known, while others are strongly tied to a particular culture.
The task is to classify items into three categories:

| Category                    | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| **Cultural Agnostic**       | Universally known items with no strong cultural association |
| **Cultural Representative** | Items originating from a culture but globally recognized    |
| **Cultural Exclusive**      | Items known primarily within a specific culture             |

This is a multi-class classification problem with subjective class boundaries and class imbalance.

---

## Approach

This approach does **not** use pretrained language models. Instead, it relies on:

* **Engineered textual features** (TF-IDF, metadata)
* **Structured knowledge** from Wikipedia and Wikidata
* **Classical machine learning models**, primarily gradient-boosted decision trees

The key insight is that **cultural specificity correlates strongly with structured encyclopedic signals**, such as language coverage, geographic properties, and cultural metadata.

---

## Model

* **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
* **Features**:

  * Wikipedia language counts and page statistics
  * Wikidata cultural and geographic properties
  * Engineered ratios and interaction features
* **Training**: Stratified train/validation split with hyperparameter tuning (Optuna)

---

## Results

**Test Set Performance (300 samples)**

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 72.67% |
| Macro F1    | 0.7066 |
| Weighted F1 | 0.7118 |

**Key Observation**
Adding Wikipedia/Wikidata features improves Macro F1 by approximately **40%** compared to a text-only baseline.

---

## Repository Structure

```
classical_ml/
├── extract_features.py        # Wikipedia/Wikidata feature extraction
├── train_classifier.py        # Model training
├── tune_hyperparameters.py   # Hyperparameter optimization
├── test_classifier.py         # Evaluation and comparison
├── models/                    # Trained models
└── results/                   # Predictions and metrics
```

---

## Notes

* This approach emphasizes **interpretability**, **efficiency**, and **explicit feature engineering**.
* It serves as a strong non-LM baseline for comparison with transformer-based models.

---

