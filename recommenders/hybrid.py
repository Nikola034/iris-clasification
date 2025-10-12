from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from configuration.config import HybridWeights
from recommenders.base import BaseRecommender
from recommenders.collaborative import CollaborativeRecommender
from recommenders.content_base import ContentRecommender
from models.sentiment import ISentimentAnalyzer, BertSentimentAnalyzer


@dataclass
class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining content-based, collaborative filtering, and sentiment analysis.
    """
    content: ContentRecommender
    cf: CollaborativeRecommender
    sentiment: ISentimentAnalyzer
    weights: HybridWeights
    content_topk: int = 200

    def __init__(self, content: ContentRecommender, cf: CollaborativeRecommender, sentiment: ISentimentAnalyzer, weights: HybridWeights, content_topk: int = 200):
        super().__init__("HybridRecommender")
        self.content, self.cf, self.sentiment, self.weights = content, cf, sentiment, weights
        self.content_topk = int(content_topk)
        self._title_index: dict[str, int] = {}
        self._titles: list[str] = []
        self._sent_prob_aligned: Optional[np.ndarray] = None

    def fit(self, df_movies: pd.DataFrame, ratings: pd.DataFrame, review_texts: list[str] | None = None, review_labels: list[int] | None = None, val_review_texts: list[str] | None = None,
            val_review_labels: list[int] | None = None) -> "HybridRecommender":

        # Fit content
        if not getattr(self.content, "fitted", False):
            self.content.fit(df_movies)
        # Titles and index to current movies
        self._titles = df_movies["clean_title"].tolist()
        self._title_index = {t: i for i, t in enumerate(self._titles)}
        self.cf.fit(ratings)

        # Train sentiment model
        if review_texts is not None and review_labels is not None and len(review_texts) and len(review_labels):
            self.sentiment.fit(review_texts, review_labels, val_texts=val_review_texts, val_labels=val_review_labels)

        # Precompute per-title sentiment probabilities
        need_sent = (self._sent_prob_aligned is None) or (len(self._sent_prob_aligned) != len(self._titles))
        if need_sent and "review" in df_movies.columns:
            # Group reviews by title
            title_col = "clean_title"
            grp = df_movies[[title_col, "review"].copy()].dropna(subset=[title_col]).groupby(title_col)["review"].apply(lambda s: [str(x) for x in s.dropna().tolist()]).to_dict()
            # Compute mean positive probability per title
            sent_probs = np.zeros(len(self._titles), dtype=float)
            has_prob = np.zeros(len(self._titles), dtype=bool)
            for title, idx in self._title_index.items():
                texts = grp.get(title, None)
                if texts and len(texts) > 0:
                    try:
                        p = self.sentiment.predict_proba(texts)[:, 1]
                        sent_probs[idx] = float(np.mean(p))
                        has_prob[idx] = True
                    except Exception:
                        has_prob[idx] = False

            fallback_titles = [self._titles[i] for i, ok in enumerate(has_prob) if not ok]
            if fallback_titles:
                try:
                    p_fallback = self.sentiment.predict_proba(fallback_titles)[:, 1]
                    j = 0
                    for i, ok in enumerate(has_prob):
                        if not ok:
                            sent_probs[i] = float(p_fallback[j])
                            j += 1
                except Exception:
                    pass

            sent_probs = np.clip(sent_probs, 0.0, 1.0)
            self._sent_prob_aligned = sent_probs

        self.fitted = True
        return self

    def _content_scores_for_seed(self, seeds):
        sims = np.zeros(len(self._titles))
        topn = int(min(len(self._titles), max(1, self.content_topk)))
        for s in seeds:
            res = self.content.recommend_for_titles([s], k=topn).get(s, [])
            for title, score in res:
                j = self._title_index.get(title, None)
                if j is not None:
                    sims[j] = max(float(sims[j]), float(score))
        return sims

    def _sentiment_scores_all(self):
        # Prefer precomputed per-title probabilities
        if self._sent_prob_aligned is not None and len(self._sent_prob_aligned) == len(self._titles):
            return self._sent_prob_aligned
        try:
            probs = self.sentiment.predict_proba(self._titles)[:, 1]
            return np.clip(probs, 0.0, 1.0)
        except Exception:
            return np.zeros(len(self._titles))

    def _cf_scores_aligned(self, user_id: str):
        # Map CF scores onto full title list.
        if not hasattr(self.cf, "item_index") or self.cf.item_index is None or not len(self.cf.item_index):
            return np.zeros(len(self._titles))
        base_scores = self.cf._score_user(user_id)
        out = np.zeros(len(self._titles))
        for title, j_cf in self.cf.item_index.items():
            j_full = self._title_index.get(title)
            if j_full is not None:
                out[j_full] = float(base_scores[j_cf])
        return out

    @staticmethod
    def _zscore(x: np.ndarray):
        if x.size == 0 or np.isnan(x).all():
            return np.zeros_like(x, dtype=float)
        m = float(np.nanmean(x))
        s = float(np.nanstd(x))
        if s < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - m) / (s + 1e-8)

    @staticmethod
    def _minmax01(x: np.ndarray):
        if x.size == 0:
            return np.zeros_like(x, dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)

    def recommend_for_user(self, user_id: str, k: int = 10, seed_titles: Iterable[str] = ()) -> list[tuple[str, float]]:
        self._assert_fitted()
        s_content = self._content_scores_for_seed(seed_titles) if seed_titles else np.zeros(len(self._titles))
        s_cf = self._cf_scores_aligned(user_id)
        s_sent = self._sentiment_scores_all()

        # Standardize each component and combine
        S_raw = self.weights.w_content * self._zscore(s_content) + self.weights.w_cf * self._zscore(s_cf) + self.weights.w_sentiment * self._zscore(s_sent)
        # Normalize ensemble
        S = self._minmax01(S_raw)

        idx = np.argsort(-S)
        recs = [(self._titles[i], float(S[i])) for i in idx[:k]]
        return recs

    def save(self, dir_path: str | Path):
        self._assert_fitted()
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        # Save components
        self.content.save(p / "content")
        self.cf.save(p / "collaborative")
        # Best-effort for sentiment analyzer
        if hasattr(self.sentiment, "save"):
            self.sentiment.save(p / "sentiment")
        # Save hybrid meta
        joblib.dump({"weights": {"w_content": self.weights.w_content, "w_cf": self.weights.w_cf, "w_sentiment": self.weights.w_sentiment}, "titles": self._titles, "title_index": self._title_index,
                     "has_sent_prob": self._sent_prob_aligned is not None, "content_topk": int(self.content_topk)}, p / "meta.pkl")
        if self._sent_prob_aligned is not None:
            joblib.dump(self._sent_prob_aligned, p / "sent_probs.npy")

    @classmethod
    def load(cls, dir_path: str | Path) -> "HybridRecommender":
        p = Path(dir_path)
        meta = joblib.load(p / "meta.pkl")
        # Load subcomponents
        content = ContentRecommender.load(p / "content")
        cf = CollaborativeRecommender.load(p / "collaborative")
        # Load sentiment
        sent_dir = p / "sentiment"
        if sent_dir.exists():
            sentiment = BertSentimentAnalyzer.load(sent_dir)
        else:
            raise FileNotFoundError("Sentiment component not found in saved hybrid model")
        weights = HybridWeights(**meta.get("weights", {}))
        content_topk = int(meta.get("content_topk", 200))
        obj = cls(content=content, cf=cf, sentiment=sentiment, weights=weights, content_topk=content_topk)
        obj._titles = list(meta.get("titles", []))
        obj._title_index = dict(meta.get("title_index", {}))
        if bool(meta.get("has_sent_prob", False)) and (p / "sent_probs.npy").exists():
            obj._sent_prob_aligned = joblib.load(p / "sent_probs.npy")
        obj.fitted = True
        return obj
