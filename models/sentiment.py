from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol, Any

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    pass