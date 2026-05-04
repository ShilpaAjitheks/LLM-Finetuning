# LLM Project....

# BERT Multi-Class Text Classification Pipeline

Fine-tuning `bert-base-cased` for multi-class text classification with a clean, modular  pipeline — covering data loading, training, evaluation, inference, and visualization in a single `.run()` call.

---
## Problem Statement

Text classification at scale requires a reliable, reproducible pipeline that handles the full ML lifecycle: from raw CSV to a saved, deployable model. This project provides an end-to-end solution for fine-tuning BERT on any multi-class text dataset, demonstrated on the [BBC News dataset](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) (5 categories: *business, entertainment, politics, sport, tech*).

---
## Architecture

| Class | Role |
|---|---|
| `Config` | All hyperparameters in one dataclass |
| `DataProcessor` | Load, encode, tokenize, build DataLoaders |
| `BERTTrainer` | Train, evaluate, predict, save |
| `Pipeline` | Orchestrates everything via `.run()` |
| `Visualiser` | Loss curves, metrics, confusion matrix |


## Dataset

Tested on [BBC News](https://www.kaggle.com/datasets/gpreda/bbc-news) (5 classes: business, entertainment, politics, sport, tech). Swap in any CSV with `text` and `label` columns.

## Results

- **Val Accuracy:** 99.3%
- **Val F1 (macro):** 99.3%

## Requirements

```
torch, transformers, datasets, scikit-learn, pandas, matplotlib, seaborn
```

GPU recommended. On CPU, lower `epochs` and `batch_size`.
