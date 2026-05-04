# LLM Project....

# BERT Multi-Class Text Classification Pipeline

Fine-tuning `bert-base-cased` for multi-class text classification with a clean, modular  pipeline — covering data loading, training, evaluation, inference, and visualization in a single `.run()` call.

---
## Problem Statement

Text classification at scale requires a reliable, reproducible pipeline that handles the full ML lifecycle: from raw CSV to a saved, deployable model. This project provides an end-to-end solution for fine-tuning BERT on any multi-class text dataset, demonstrated on the [BBC News dataset](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) (5 categories: *business, entertainment, politics, sport, tech*).

## Architecture

The pipeline is decomposed into five single-responsibility classes:

| Class | Responsibility |
|---|---|
| `Config` | Centralized hyperparameters via `@dataclass` — single source of truth |
| `DataProcessor` | CSV/Excel loading, class filtering, label encoding, tokenization, DataLoader creation |
| `BERTTrainer` | Model init, optimizer/scheduler setup, training loop, evaluation, inference, saving |
| `Pipeline` | Orchestrates all 6 steps via a single `.run()` call |
| `Visualiser` | Loss curves, validation metrics, and confusion matrix plots from a trained `Pipeline` |

**Training flow:**

```
Raw CSV → DataProcessor → (train/test split, tokenization) → DataLoaders
         → BERTTrainer → (fine-tune, per-epoch eval) → Classification Report + Saved Model
         → Visualiser  → (loss curve, F1/accuracy curve, confusion matrix)
```

---
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
