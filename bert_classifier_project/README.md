# Bert-classifier

Fine-tune BERT (or any HuggingFace encoder) for multi-class text classification — packaged into a clean, installable Python project with a CLI and tests.

## What it does

Takes a CSV of `(text, label)` rows, fine-tunes a BERT model on it, evaluates on a held-out split, saves the trained model + classification report + zip archive, and (optionally) plots loss / accuracy / F1 curves and a confusion matrix.

## Project layout

```
.
├── src/bert_classifier/      # the package
│   ├── config.py             # Config dataclass — all hyperparameters
│   ├── data.py               # CSV/Excel loading, tokenization, DataLoaders
│   ├── trainer.py            # model setup, training, evaluation, inference, save
│   ├── pipeline.py           # orchestrator — wires it all together
│   ├── visualizer.py         # learning curves + confusion matrix
│   └── cli.py                # `python main.py` entry point
├── tests/                    # pytest unit tests
├── main.py                   # CLI launcher
├── pyproject.toml            # package metadata + deps
└── requirements.txt
```

## Setup

Requires Python 3.9+. Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv

# Activate it:
source .venv/bin/activate            # macOS / Linux
source .venv/Scripts/activate        # Windows (Git Bash)
.\.venv\Scripts\Activate.ps1         # Windows (PowerShell)

pip install -e ".[dev]"
```

The last command installs all runtime dependencies (torch, transformers, pandas, ...) plus pytest in editable mode, so any code changes take effect immediately without reinstalling.

## Usage

Place your data file (default name `bbc_news_full.csv`) in the project root, then:

```bash
# Train + evaluate + save the model
python main.py --epochs 3 --batch-size 16

# Use a smaller / different backbone
python main.py --checkpoint distilbert-base-cased --epochs 3

# Train on a subset of classes
python main.py --classes business sport politics

# Train then predict on raw text
python main.py --predict "The central bank raised rates today." \
                "The team won the championship in dramatic fashion."

# Save training_results.png after training
python main.py --epochs 3 --plot
```

Run `python main.py --help` to see every option.

## Programmatic use

```python
from bert_classifier import Config
from bert_classifier.pipeline import Pipeline

cfg = Config(data_path="my.csv", epochs=5, optimizer="AdamW", lr_scheduler="cosine")
pipeline = Pipeline(cfg).run()

print(pipeline.predict(["some new text", "another one"]))
```

## Output

After a run, you'll find:

```
bert_multiclass_model/
├── config.json
├── model.safetensors
├── tokenizer files
├── label_encoder.pkl
└── classification_report.txt
bert_multiclass_model.zip   ← ready to share / deploy
training_results.png         ← only if --plot
```

## Tests

```bash
pytest
```

13 tests covering config defaults, data loading, label encoding, optimizer/scheduler validation, pipeline error handling, and CLI parsing.

## Configuration reference

| Option              | Default              | Notes                                              |
|---------------------|----------------------|----------------------------------------------------|
| `--data-path`       | `bbc_news_full.csv`  | CSV or Excel file                                  |
| `--text-col`        | `text`               | column name for input text                         |
| `--label-col`       | `label_text`         | column name for labels                             |
| `--checkpoint`      | `bert-base-cased`    | any HuggingFace encoder                            |
| `--epochs`          | `3`                  |                                                    |
| `--batch-size`      | `16`                 |                                                    |
| `--lr`              | `2e-5`               | learning rate                                      |
| `--optimizer`       | `AdamW`              | `AdamW` / `RMSprop` / `Adagrad` / `Adadelta`       |
| `--scheduler`       | `linear`             | `linear` / `cosine` / `polynomial` / `exponential` |
| `--save-dir`        | `bert_multiclass_model` |                                                 |
| `--classes`         | (all)                | space-separated subset of class names              |
| `--predict`         | —                    | space-separated texts to classify after training   |
| `--plot`            | off                  | generate `training_results.png`                    |

## Notes

- GPU is auto-detected via `torch.cuda.is_available()`. CPU works but is slow.
- Random seed is fixed at `0` for reproducibility.
- Tested on Python 3.11 (recommended). Python 3.13 may segfault on Windows due to torch instability — downgrade if you hit that.
