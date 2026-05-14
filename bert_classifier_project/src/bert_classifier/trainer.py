import os
import pickle
import shutil
import numpy as np

import torch
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, accuracy_score, f1_score

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adagrad, Adadelta, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    PolynomialLR,
    ExponentialLR,
)

from .config import Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTTrainer:
    """Train, evaluate, and run inference with a BERT classification model."""

    OPTIMIZERS = {
        "AdamW":    AdamW,
        "RMSprop":  RMSprop,
        "Adagrad":  Adagrad,
        "Adadelta": Adadelta,
    }

    def __init__(self, cfg: Config, num_classes: int):
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.checkpoint, num_labels=num_classes,
        ).to(DEVICE)
        self.history = []

    # ── Setup ──

    def setup(self, train_dl):
        """
        Configure optimizer and LR scheduler based on Config.

        Must be called before train().
        """
        self.total_steps = self.cfg.epochs * len(train_dl)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        print(f"  Optimizer: {self.cfg.optimizer}  |  Scheduler: {self.cfg.lr_scheduler}")
        print(f"  Total training steps: {self.total_steps:,}")

    # ── Training ──

    def train(self, train_dl, eval_dl=None, class_names=None):
        progress = tqdm(range(self.total_steps), desc="Training")

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = self.model(**batch)

                outputs.loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += outputs.loss.item()
                progress.update(1)

            train_loss = epoch_loss / len(train_dl)
            record = {"epoch": epoch, "train_loss": train_loss}

            # Validation after each epoch
            if eval_dl is not None:
                val_metrics = self._compute_metrics(eval_dl)
                record.update(val_metrics)
                print(f"  Epoch {epoch}/{self.cfg.epochs}  |  "
                      f"train_loss = {train_loss:.4f}  |  "
                      f"val_loss = {val_metrics['val_loss']:.4f}  |  "
                      f"val_acc = {val_metrics['val_acc']:.4f}  |  "
                      f"val_f1 = {val_metrics['val_f1']:.4f}")
            else:
                print(f"  Epoch {epoch}/{self.cfg.epochs}  |  train_loss = {train_loss:.4f}")

            self.history.append(record)

    # ── Evaluation ──

    def evaluate(self, eval_dl, class_names):
        """Run inference on eval set and return a classification report string."""
        preds, trues, _ = self._run_inference(eval_dl)
        self.y_true, self.y_pred = trues, preds

        return classification_report(
            trues, preds,
            target_names=class_names,
            zero_division=0,
        )

    # ── Inference ──

    def predict(self, texts, data_processor):
        """Predict class labels for a list of raw text strings."""
        encoded = {}
        for text in texts:
            tok = data_processor.tokenizer(text, truncation=True)
            for key in tok:
                encoded.setdefault(key, []).append(tok[key])

        ds = Dataset.from_dict(encoded)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, collate_fn=data_processor.collator)

        preds, _, _ = self._run_inference(dl, collect_labels=False)
        return data_processor.class_names[preds]

    # ── Save ──

    def save(self, report: str, tokenizer, label_encoder=None):
        """
        Save model weights, tokenizer, label encoder, classification report,
        and create a zip archive.
        """
        save_dir = self.cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        if label_encoder is not None:
            with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(label_encoder, f)

        shutil.make_archive(save_dir, "zip", save_dir)

        print(f"  Model saved → {save_dir}/")
        print(f"  Archive     → {save_dir}.zip")

    # ── Private helpers ──

    def _build_optimizer(self):
        name = self.cfg.optimizer
        if name not in self.OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {name}. Choose from {list(self.OPTIMIZERS)}")
        return self.OPTIMIZERS[name](self.model.parameters(), lr=self.cfg.learning_rate)

    def _build_scheduler(self):
        schedulers = {
            "linear":      lambda: get_scheduler("linear", optimizer=self.optimizer,
                                                 num_warmup_steps=4,
                                                 num_training_steps=self.total_steps),
            "cosine":      lambda: CosineAnnealingLR(self.optimizer, T_max=self.total_steps),
            "polynomial":  lambda: PolynomialLR(self.optimizer, total_iters=self.total_steps, power=2),
            "exponential": lambda: ExponentialLR(self.optimizer, gamma=0.1),
        }
        name = self.cfg.lr_scheduler
        if name not in schedulers:
            raise ValueError(f"Unsupported scheduler: {name}. Choose from {list(schedulers)}")
        return schedulers[name]()

    @torch.no_grad()
    def _run_inference(self, dl, collect_labels=True, collect_loss=False):
        """
        Shared inference loop used by _compute_metrics, evaluate, and predict.

        Args:
            dl: DataLoader to run through the model.
            collect_labels: If True, also return true labels (needs "labels" in batch).
            collect_loss:   If True, also accumulate and return average loss.

        Returns:
            (preds, trues, avg_loss) — trues / avg_loss are empty/None if not collected.
        """
        self.model.eval()
        all_preds, all_trues = [], []
        total_loss = 0.0

        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = self.model(**batch)

            all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())

            if collect_loss:
                total_loss += outputs.loss.item() * batch["labels"].size(0)
            if collect_labels and "labels" in batch:
                all_trues.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(dl.dataset) if collect_loss else None
        return np.array(all_preds), np.array(all_trues), avg_loss

    def _compute_metrics(self, eval_dl):
        """Used during training — returns numeric metrics for logging."""
        preds, trues, val_loss = self._run_inference(eval_dl, collect_loss=True)
        return {
            "val_loss": val_loss,
            "val_acc":  accuracy_score(trues, preds),
            "val_f1":   f1_score(trues, preds, average="macro", zero_division=0),
        }
