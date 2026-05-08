# LLM Project....

# BERT Multi-Class Text Classification Pipeline

Fine-tuning `bert-base-cased` for multi-class text classification with a clean, modular  pipeline — covering data loading, training, evaluation, inference, and visualization in a single `.run()` call.

---
## Problem Statement

Text classification at scale requires a reliable, reproducible pipeline that handles the full ML lifecycle: from raw CSV to a saved, deployable model. This project provides an end-to-end solution for fine-tuning any HuggingFace encoder on any multi-class text dataset, demonstrated on the [BBC News dataset](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) (5 categories: *business, entertainment, politics, sport, tech*).


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

---

## Key Results

- **Val Accuracy:** 
- **Val F1 (macro):** 

---

## Setup & Installation

### Requirements

- Python 3.8+
- GPU recommended (CUDA). CPU is supported but slow.

### Install dependencies

```bash
pip install transformers datasets scikit-learn torch pandas openpyxl evaluate accelerate tqdm matplotlib seaborn
```
Or in Colab/Jupyter:

```python
!pip install -q transformers datasets scikit-learn torch pandas openpyxl evaluate accelerate tqdm
```
### Dataset

Download `bbc_news_full.csv` and place it in the working directory. The file must contain at minimum:
- a text column (default: `text`)
- a label column (default: `label_text`)

---
## Usage

### Full pipeline (one call)

```python
from bert_multiclass_pipeline_final import Config, Pipeline

cfg = Config()          # all defaults
pipeline = Pipeline(cfg).run()
```
### Customize hyperparameters

```python
cfg = Config(
    data_path="your_data.csv",
    text_col="content",
    label_col="category",
    epochs=5,
    batch_size=32,
    learning_rate=3e-5,
    optimizer="AdamW",          # AdamW | RMSprop | Adagrad | Adadelta
    lr_scheduler="cosine",      # linear | cosine | polynomial | exponential
)
pipeline = Pipeline(cfg).run()
```
### Train on a class subset

```python
pipeline = Pipeline(cfg).run(allowed_classes=["sport", "business"])
```

### Inference on new text

```python
predictions = pipeline.predict([
    "The quarterly revenue exceeded analyst expectations by 12%.",
    "New data privacy regulations were signed into law yesterday.",
])
print(predictions)  # e.g. ['business', 'politics']
```
### Visualize results

```python
from bert_multiclass_pipeline_final import Visualiser

viz = Visualiser(pipeline)
viz.plot_all()              # 3-panel: loss, metrics, confusion matrix
viz.plot_loss()
viz.plot_metrics()
viz.plot_confusion_matrix()
```

---

## Output Artifacts

After a successful run, the following are saved to `bert_multiclass_model/` (configurable via `Config.save_dir`):

```
bert_multiclass_model/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── vocab.txt
├── label_encoder.pkl
└── classification_report.txt

bert_multiclass_model.zip   ← ready for download / deployment
```

## Notes

- **Swap backbone:** Any HuggingFace encoder works — `roberta-base`, `distilbert-base-cased`, `albert-base-v2`, etc.
- **Reproducibility:** Seed is fixed at `0` for `torch`, `numpy`, and `train_test_split`.
- **Access internals:** `pipeline.report`, `pipeline.trainer.model`, `pipeline.data_processor.class_names`
- **GPU:** Automatically detected via `torch.cuda.is_available()`.
