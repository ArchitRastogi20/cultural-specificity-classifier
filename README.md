# Cultural Specificity Classifier

This repository contains two complementary approaches for classifying items based on their degree of cultural specificity. The task involves assigning each item to one of three classes: Cultural Agnostic, Cultural Representative, or Cultural Exclusive.

The two approaches differ in methodology, modeling assumptions, and feature representations.

---

## Repository Structure

```
.
├── classical_ml/
├── transformer_based/
└── README.md
```

---

## Approaches

### Classical Machine Learning (Non-LM)

The `classical_ml/` directory contains a **non–language-model-based** solution built using classical machine learning techniques. This approach relies on engineered features derived from textual attributes and external structured knowledge (e.g., Wikipedia and Wikidata metadata), combined with traditional classifiers such as gradient-boosted decision trees.

This approach emphasizes interpretability, computational efficiency, and explicit feature engineering.

Refer to `classical_ml/README.md` for detailed methodology, experiments, and results.

---

### Transformer-Based Models (LM)

The `transformer_based/` directory contains a **language-model-based** solution leveraging pretrained transformer architectures. This approach fine-tunes multilingual transformer models on the cultural specificity classification task, using learned contextual representations instead of manually engineered features.

This approach focuses on representation learning and end-to-end neural modeling.

Refer to `transformer_based/README.md` for detailed methodology, training setup, and evaluation results.

---

## Notes

* Both approaches are self-contained and independently reproducible.
* The repository is structured to allow direct comparison between classical (non-LM) and transformer-based (LM) methodologies for the same task and dataset.

---

## License

This project is released under the Apache 2.0 License.

---
