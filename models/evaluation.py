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

def evaluate_recommender_loo(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, rec_builder: Callable[[], object], ks: Iterable[int] = (5, 10), rating_threshold: int = 7, max_users: Optional[int] = None,
                             max_heldout_per_user: Optional[int] = None, user_sample_seed: Optional[int] = None):
    """
    Evaluate a recommender using leave-one-out (LOO) protocol over users with >=7 ratings.
    :param movies_df:
    :param ratings_df:
    :param rec_builder:
    :param ks:
    :param rating_threshold:
    :param max_users:
    :param max_heldout_per_user:
    :param user_sample_seed:
    :return:
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    max_k = max(ks) if ks else 10

    # Accumulators
    prec_sums = {k: 0.0 for k in ks}
    rec_sums = {k: 0.0 for k in ks}
    map_sums = {k: 0.0 for k in ks}
    n_rank_cases = 0

    y_true_rmse: List[float] = []
    y_pred_rmse: List[float] = []

    # Users with >= 2 ratings
    users = [u for u, grp in ratings_df.groupby("user_id") if len(grp) >= 2]
    if max_users is not None:
        if user_sample_seed is not None:
            users = list(users)
            np.random.shuffle(users)
        users = users[:max_users]

    for user_id in users:
        user_r = ratings_df[ratings_df["user_id"] == user_id].reset_index(drop=True)
        heldout_indices = list(range(len(user_r)))
        if max_heldout_per_user is not None and max_heldout_per_user < len(heldout_indices):
            np.random.shuffle(heldout_indices)
            heldout_indices = heldout_indices[:max_heldout_per_user]

        # For content recommender, build once per user to avoid repeated fits
        first_rec = rec_builder()
        use_single_rec = isinstance(first_rec, ContentRecommender)
        if use_single_rec:
            rec_user = first_rec
            if not getattr(rec_user, "fitted", False):
                rec_user.fit(movies_df)
        else:
            rec_user = None

        for i in heldout_indices:
            test_row = user_r.iloc[i]
            # Extract scalar values explicitly
            title_str = str(test_row.get("movie_title"))
            rating_val = float(np.asarray(test_row.get("rating")).item()) if "rating" in test_row else float(test_row.get("rating", 0.0))

            train_mask = ~(ratings_df["user_id"] == user_id) & (ratings_df["movie_title"] == title_str) & (ratings_df["rating"] == rating_val)
            train_r = ratings_df[train_mask]

            # Build and fit recommender on training ratings
            if use_single_rec:
                rec = rec_user
            else:
                rec = first_rec if (i == heldout_indices[0]) else rec_builder()

            if isinstance(rec, CollaborativeRecommender):
                rec.fit(train_r)
            elif isinstance(rec, ContentRecommender):
                # Already fitted above if needed
                pass
            elif isinstance(rec, HybridRecommender):
                # Note: we don't (re)train sentiment inside here; Hybrid will precompute sentiments over movies.
                rec.fit(movies_df, train_r)
            else:
                # Unknown type; skip
                continue

            # Prepare recommendation list for ranking metrics
            rec_titles: List[str] = []
            if isinstance(rec, CollaborativeRecommender):
                rec_titles = [t for t, _ in rec.recommend_for_user(user_id=user_id, k=max_k)]
            elif isinstance(rec, HybridRecommender):
                seeds = _choose_seed_titles(train_r[train_r["user_id"] == user_id], top_n=1)
                rec_titles = [t for t, _ in rec.recommend_for_user(user_id=user_id, k=max_k, seed_titles=seeds)]
            elif isinstance(rec, ContentRecommender):
                seeds = _choose_seed_titles(train_r[train_r["user_id"] == user_id], top_n=1)
                if seeds:
                    res = rec.recommend_for_titles(seeds, k=max_k)
                    rec_titles = [t for t, _ in res.get(seeds[0], [])]
                else:
                    rec_titles = []

            # Ranking metrics with binary relevance on held-out rating
            if rating_val >= float(rating_threshold):
                relevant = {title_str}
                for k in ks:
                    prec_sums[k] += RankingMetrics.precision_at_k(rec_titles, relevant, k)
                    rec_sums[k] += RankingMetrics.recall_at_k(rec_titles, relevant, k)
                    map_sums[k] += RankingMetrics.average_precision(rec_titles, relevant, k)
                n_rank_cases += 1

            # RMSE over numeric predictions where supported (CF)
            if isinstance(rec, CollaborativeRecommender):
                # Predict rating for held-out item
                scores = rec._score_user(user_id)
                j = rec.item_index.get(title_str)
                if j is not None and 0 <= j < len(scores):
                    y_pred = float(scores[j])
                else:
                    y_pred = float(rec.global_mean)
                y_true_rmse.append(rating_val)
                y_pred_rmse.append(y_pred)

    # Aggregate
    out = {}
    for k in ks:
        out[f"precision@{k}"] = (prec_sums[k] / n_rank_cases) if n_rank_cases else 0.0
        out[f"recall@{k}"] = (rec_sums[k] / n_rank_cases) if n_rank_cases else 0.0
        out[f"map@{k}"] = (map_sums[k] / n_rank_cases) if n_rank_cases else 0.0

    if y_true_rmse:
        out["rmse"] = RatingMetrics.rmse(np.array(y_true_rmse, dtype=float), np.array(y_pred_rmse, dtype=float))
    else:
        out["rmse"] = 0.0

    out["n_rank_cases"] = float(n_rank_cases)
    out["n_users"] = float(len(users))
    return out


def ab_test_simulation(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, builders: Dict[str, Callable[[], object]], ks: Iterable[int] = (5, 10), rating_threshold: int = 7, max_users: Optional[int] = None,
                       max_heldout_per_user: Optional[int] = None, user_sample_seed: Optional[int] = None):
    """
    A/B test simulation comparing multiple recommenders using leave-one-out evaluation.
    :param movies_df:
    :param ratings_df:
    :param builders:
    :param ks:
    :param rating_threshold:
    :param max_users:
    :param max_heldout_per_user:
    :param user_sample_seed:
    :return:
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, builder in builders.items():
        metrics = evaluate_recommender_loo(movies_df, ratings_df, builder, ks=ks, rating_threshold=rating_threshold, max_users=max_users, max_heldout_per_user=max_heldout_per_user, user_sample_seed=user_sample_seed)
        results[name] = metrics
    return results


def qualitative_topk_for_users(ratings_df: pd.DataFrame, recs: Dict[str, object], sample_users: Iterable[str], k: int = 10):
    """
    Generate top-k recommendations for a few sample users from different recommenders for qualitative inspection.
    :param ratings_df:
    :param recs:
    :param sample_users:
    :param k:
    :return:
    """
    out = {}
    for user in sample_users:
        out[user] = {}
        user_train = ratings_df[ratings_df["user_id"] == user]
        seeds = _choose_seed_titles(user_train, top_n=1)
        for name, rec in recs.items():
            if isinstance(rec, CollaborativeRecommender):
                out[user][name] = rec.recommend_for_user(user, k=k, normalize=True)
            elif isinstance(rec, HybridRecommender):
                out[user][name] = rec.recommend_for_user(user, k=k, seed_titles=seeds)
            elif isinstance(rec, ContentRecommender):
                if seeds:
                    res = rec.recommend_for_titles(seeds, k=k)
                    out[user][name] = res.get(seeds[0], [])
                else:
                    out[user][name] = []
            else:
                out[user][name] = []
    return out
