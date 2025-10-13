from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from recommenders.base import BaseRecommender
from utils.utils import to_user_item_matrix


@dataclass
class CollaborativeRecommender(BaseRecommender):
    algo: str = "svd"
    n_factors: int = 50
    reg: float = 0.02
    n_iters: int = 20

    def __init__(self, algo: str = "svd", n_factors: int = 50, reg: float = 0.02, n_iters: int = 20):
        super().__init__("CollaborativeRecommender")
        self.algo, self.n_factors, self.reg, self.n_iters = algo, n_factors, reg, n_iters
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.U = None
        self.V = None
        self.global_mean = 0.0
        self.rating_min: float | None = None
        self.rating_max: float | None = None

    def fit(self, ratings: pd.DataFrame, user_col: str = "user_id", item_col: str = "movie_title", rating_col: str = "rating") -> "CollaborativeRecommender":
        mat, users, items = to_user_item_matrix(ratings, user_col, item_col, rating_col)
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {m: j for j, m in enumerate(items)}

        # Rating scale bounds for normalization
        if rating_col in ratings.columns and not ratings.empty:
            self.rating_min = float(ratings[rating_col].min())
            self.rating_max = float(ratings[rating_col].max())
        else:
            self.rating_min, self.rating_max = 1.0, 5.0

        # Centering
        finite_mask = np.isfinite(mat)
        self.global_mean = float(np.nanmean(mat[finite_mask])) if finite_mask.any() else 0.0
        R = np.where(finite_mask, mat - self.global_mean, 0.0)
        mask = finite_mask.astype(float)

        # Initialize factors
        U = np.random.normal(scale=0.1, size=(R.shape[0], self.n_factors))
        V = np.random.normal(scale=0.1, size=(R.shape[1], self.n_factors))

        # Simple ALS-like alternating updates
        lam = self.reg
        for _ in range(self.n_iters):
            # Update U
            for i in range(R.shape[0]):
                M_i = np.diag(mask[i])
                A = V.T @ M_i @ V + lam * np.eye(self.n_factors)
                b = V.T @ M_i @ R[i]
                U[i] = np.linalg.solve(A, b)
            # Update V
            for j in range(R.shape[1]):
                M_j = np.diag(mask[:, j])
                A = U.T @ M_j @ U + lam * np.eye(self.n_factors)
                b = U.T @ M_j @ R[:, j]
                V[j] = np.linalg.solve(A, b)

        self.U, self.V = U, V
        self.fitted = True
        return self

    def _score_user(self, user_id: str) -> np.ndarray:
        self._assert_fitted()
        if user_id not in self.user_index:
            return np.zeros(len(self.item_index))
        i = self.user_index[user_id]
        scores = self.U[i] @ self.V.T + self.global_mean
        return scores

    def _normalize_scores01(self, scores: np.ndarray) -> np.ndarray:
        lo = self.rating_min if self.rating_min is not None else 1.0
        hi = self.rating_max if self.rating_max is not None else 5.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 1.0, 5.0
        return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)

    def recommend_for_user(self, user_id: str, k: int = 10, normalize: bool = False) -> list[tuple[str, float]]:
        scores = self._score_user(user_id)
        if normalize:
            scores = self._normalize_scores01(scores)
        seen = set()
        titles = list(self.item_index.keys())
        idx = np.argsort(-scores)
        recs = []
        for j in idx:
            t = titles[j]
            if t in seen:
                continue
            recs.append((t, float(scores[j])))
            if len(recs) >= k:
                break
        return recs

    def save(self, dir_path: str | Path):
        self._assert_fitted()
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump({"algo": self.algo, "n_factors": self.n_factors, "reg": self.reg, "n_iters": self.n_iters, "user_index": self.user_index, "item_index": self.item_index, "global_mean": self.global_mean,
                     "rating_min": self.rating_min, "rating_max": self.rating_max}, p / "meta.pkl")
        joblib.dump(self.U, p / "U.pkl")
        joblib.dump(self.V, p / "V.pkl")

    @classmethod
    def load(cls, dir_path: str | Path) -> "CollaborativeRecommender":
        p = Path(dir_path)
        meta = joblib.load(p / "meta.pkl")
        obj = cls(algo=meta.get("algo", "svd"), n_factors=int(meta.get("n_factors", 50)), reg=float(meta.get("reg", 0.02)), n_iters=int(meta.get("n_iters", 20)))
        obj.user_index = dict(meta.get("user_index", {}))
        obj.item_index = dict(meta.get("item_index", {}))
        obj.global_mean = float(meta.get("global_mean", 0.0))
        obj.rating_min = float(meta.get("rating_min", 1.0)) if meta.get("rating_min", None) is not None else None
        obj.rating_max = float(meta.get("rating_max", 5.0)) if meta.get("rating_max", None) is not None else None
        obj.U = joblib.load(p / "U.pkl")
        obj.V = joblib.load(p / "V.pkl")
        obj.fitted = True
        return obj
