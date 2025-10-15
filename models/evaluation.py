from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from configuration.config import SentimentConfig, SystemConfig
from recommenders.collaborative import CollaborativeRecommender
from recommenders.content_base import ContentRecommender
from recommenders.hybrid import HybridRecommender
from models.sentiment import build_sentiment_analyzer, ISentimentAnalyzer


@dataclass
class RankingMetrics:
    """
    Ranking metrics for recommendation lists.
    """

    @staticmethod
    def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if k == 0:
            return 0.0
        return len([i for i in recommended[:k] if i in relevant]) / k

    @staticmethod
    def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if len(relevant) == 0:
            return 0.0
        hit = len([i for i in recommended[:k] if i in relevant])
        return hit / len(relevant)

    @staticmethod
    def average_precision(recommended: list[str], relevant: set[str], k: int) -> float:
        ap, hits = 0.0, 0
        for i, item in enumerate(recommended[:k], start=1):
            if item in relevant:
                hits += 1
                ap += hits / i
        return ap / max(1, len(relevant))

    @staticmethod
    def mean_average_precision(all_recs: list[list[str]], all_rels: list[set[str]], k: int) -> float:
        aps = [RankingMetrics.average_precision(r, rel, k) for r, rel in zip(all_recs, all_rels)]
        return float(np.mean(aps)) if aps else 0.0


@dataclass
class RatingMetrics:
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class SentimentMetrics:
    """Binary classification metrics for sentiment analysis."""

    @staticmethod
    def compute(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
        y_true = np.asarray(list(y_true), dtype=int)
        y_pred = np.asarray(list(y_pred), dtype=int)
        acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def cross_validate_sentiment_kfold(texts: Iterable[str], labels: Iterable[int], cfg: SentimentConfig, system_cfg: SystemConfig, k=5, seed=19):
    """
    Stratified K-Fold cross-validation for sentiment analysis.
    :param texts:
    :param labels:
    :param cfg:
    :param system_cfg:
    :param k:
    :param seed:
    :return:
    """
    X = np.array(list(texts), dtype=object)
    y = np.array(list(labels), dtype=int)
    if X.size == 0:
        return {"folds": [], "mean": {m: 0.0 for m in ["accuracy", "precision", "recall", "f1"]}}

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []

    for tr_idx, te_idx in skf.split(X, y):
        model: ISentimentAnalyzer = build_sentiment_analyzer(cfg, system_cfg)
        x_tr, y_tr = X[tr_idx].tolist(), y[tr_idx].tolist()
        x_te, y_te = X[te_idx].tolist(), y[te_idx].tolist()
        # Small validation split from train
        if len(x_tr) >= 5:
            n_val = max(1, int(0.2 * len(x_tr)))
            x_val, y_val = x_tr[:n_val], y_tr[:n_val]
            x_tr2, y_tr2 = x_tr[n_val:], y_tr[n_val:]
        else:
            x_val, y_val = [], []
            x_tr2, y_tr2 = x_tr, y_tr

        model.fit(x_tr2, y_tr2, val_texts=x_val, val_labels=y_val)
        y_pred = model.predict(x_te)
        m = SentimentMetrics.compute(y_te, y_pred)
        fold_metrics.append(m)

    # Aggregate mean metrics
    mean_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) if fold_metrics else 0.0 for k in ["accuracy", "precision", "recall", "f1"]}
    return {"folds": fold_metrics, "mean": mean_metrics}


def _choose_seed_titles(train_ratings: pd.DataFrame, top_n: int = 1) -> list[str]:
    if train_ratings is None or train_ratings.empty:
        return []
    g = train_ratings.sort_values(by=["rating"], ascending=False)
    titles = g["movie_title"].astype(str).tolist()
    # Deduplicate while preserving order
    seen = set()
    ordered = [t for t in titles if not (t in seen or seen.add(t))]
    return ordered[:top_n]