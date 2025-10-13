from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configuration.config import SentimentConfig, SystemConfig
from utils.utils import clean_text


class ISentimentAnalyzer(Protocol):
    def fit(self, texts: Iterable[str], labels: Iterable[int], val_texts: Iterable[str] = None, val_labels: Iterable[int] = None) -> 'ISentimentAnalyzer': ...

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray: ...

    def predict(self, texts: Iterable[str]) -> np.ndarray: ...

    def save(self, dir_path: str | Path) -> None: ...


class TextDS(Dataset):
    def __init__(self, x, y, tok, max_len):
        self.x, self.y, self.tok, self.max_len = list(x), list(y), tok, max_len

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        enc = self.tok(clean_text(self.x[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return item


class TextDSPred(Dataset):
    def __init__(self, x, tok, max_len):
        self.x, self.tok, self.max_len = list(x), tok, max_len

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        enc = self.tok(clean_text(self.x[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


@dataclass
class BertSentimentAnalyzer(ISentimentAnalyzer):
    """
    Simple BERT-based sentiment analyzer using HuggingFace transformers.
    """
    cfg: SentimentConfig
    system_cfg: SystemConfig
    _model: Any = None
    _tokenizer: Any = None

    def _setup(self):
        self._device = self.system_cfg.device
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_name, num_labels=2)
        self._model.to(self._device)

    def fit(self, texts: Iterable[str], labels: Iterable[int], val_texts: Iterable[str] = None, val_labels: Iterable[int] = None) -> 'ISentimentAnalyzer':
        if self._model is None:
            self._setup()
        ds = TextDS(texts, labels, self._tokenizer, self.cfg.max_len)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_dl = None
        if val_texts and val_labels:
            val_ds = TextDS(val_texts, val_labels, self._tokenizer, self.cfg.max_len)
            val_dl = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        opt = AdamW(self._model.parameters(), lr=self.cfg.lr)

        # Training loop
        for epoch in range(self.cfg.epochs):
            self._model.train()
            total_loss = 0.0
            for batch in dl:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs = self._model(**batch)
                loss = outputs.loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl) if len(dl) > 0 else 0.0
            print(f"Epoch {epoch + 1}/{self.cfg.epochs}, Loss: {avg_loss:.4f}")

            if val_dl is None:
                continue
            # Validation
            self._model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_dl:
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    outputs = self._model(**batch)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += batch["labels"].size(0)
            val_acc = correct / total if total > 0 else 0.0
            print(f"Validation Accuracy: {val_acc:.4f}")
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        if self._model is None:
            self._setup()
        ds = TextDSPred(texts, self._tokenizer, self.cfg.max_len)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False)
        self._model.eval()
        probs = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = self._model(**batch).logits
                p = F.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        return np.vstack(probs) if probs else np.zeros((0, 2), dtype=float)

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)

    # Save and load using HF transformers save_pretrained and from_pretrained
    def save(self, dir_path: str | Path) -> None:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        # Save HF model and tokenizer
        model_dir = p / "hf_model"
        tok_dir = p / "hf_tokenizer"
        model_dir.mkdir(exist_ok=True)
        tok_dir.mkdir(exist_ok=True)
        if self._model is None or self._tokenizer is None:
            self._setup()
        # Use transformers save_pretrained
        self._model.save_pretrained(str(model_dir))
        self._tokenizer.save_pretrained(str(tok_dir))
        # Save configs
        with (p / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"cfg": asdict(self.cfg), "system_cfg": asdict(self.system_cfg)}, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> 'BertSentimentAnalyzer':
        p = Path(dir_path)
        with (p / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = SentimentConfig(**meta.get("cfg", {}))
        system_cfg = SystemConfig(**meta.get("system_cfg", {}))
        obj = cls(cfg=cfg, system_cfg=system_cfg)

        obj._device = system_cfg.device
        obj._tokenizer = AutoTokenizer.from_pretrained(str(p / "hf_tokenizer"))
        obj._model = AutoModelForSequenceClassification.from_pretrained(str(p / "hf_model"))
        obj._model.to(obj._device)
        return obj


def build_sentiment_analyzer(cfg: SentimentConfig, system_cfg: SystemConfig) -> ISentimentAnalyzer:
    return BertSentimentAnalyzer(cfg=cfg, system_cfg=system_cfg)
