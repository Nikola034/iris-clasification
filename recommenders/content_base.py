from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommenders.base import BaseRecommender
from utils.utils import clean_text


@dataclass
class ContentRecommender(BaseRecommender):
    max_features: int = 30000

    def __init__(self, max_features: int = 30000):
        super().__init__(name="ContentRecommender")
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf = None
        self._titles: list[str] = []

    def fit(self, df: pd.DataFrame, text_cols: Iterable[str] = ("clean_keywords", "clean_genres", "clean_title")) -> "ContentRecommender":
        texts = df.loc[:, list(text_cols)].fillna("").agg(lambda r: " ".join(clean_text(str(v)) for v in r if v), axis=1)
        self._vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), min_df=2, stop_words='english')
        self._tfidf = self._vectorizer.fit_transform(texts.tolist())
        self._titles = df["clean_title"].fillna("").tolist()
        self.fitted = True
        return self

    def _sim_scores_for_title(self, title, top_k):
        self._assert_fitted()
        title = title.strip().lower()
        lower_titles = [t.lower() for t in self._titles]
        if title not in lower_titles:
            return []
        idx = lower_titles.index(title)
        sims = cosine_similarity(self._tfidf[idx], self._tfidf).ravel()
        sims[idx] = -1  # exclude self
        top_k = min(top_k, len(sims) - 1)
        if top_k <= 0:
            return []
        part = np.argpartition(-sims, top_k - 1)[:top_k]
        pairs = [(self._titles[i], float(sims[i])) for i in part]
        pairs.sort(key=lambda x: -x[1])
        return pairs

    def recommend_for_titles(self, titles, k=10):
        return {t: self._sim_scores_for_title(t, k) for t in titles}

    def save(self, dir_path: str | Path):
        """Persist vectorizer, tfidf matrix, titles and config to a folder."""
        self._assert_fitted()
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "max_features": self.max_features,
            "titles": self._titles,
        }, p / "meta.pkl")
        joblib.dump(self._vectorizer, p / "vectorizer.pkl")
        joblib.dump(self._tfidf, p / "tfidf.pkl")

    @classmethod
    def load(cls, dir_path: str | Path) -> "ContentRecommender":
        p = Path(dir_path)
        meta = joblib.load(p / "meta.pkl")
        obj = cls(max_features=int(meta.get("max_features", 30000)))
        obj._vectorizer = joblib.load(p / "vectorizer.pkl")
        obj._tfidf = joblib.load(p / "tfidf.pkl")
        obj._titles = list(meta.get("titles", []))
        obj.fitted = True
        return obj
