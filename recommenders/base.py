from __future__ import annotations


class BaseRecommender:
    def __init__(self, name: str):
        self.name = name
        self.fitted: bool = False

    def _assert_fitted(self):
        if not self.fitted:
            raise RuntimeError(f"{self.name} not fitted yet.")
