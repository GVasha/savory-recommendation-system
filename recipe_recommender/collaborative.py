from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .core import RecipeRecommender


class ItemItemCollaborativeFiltering:
    """
    Memory-based item-item CF from implicit co-occurrence within user histories.
    For each user train list, increments co-occurrence counts between distinct items.
    Recommendation score for candidate items is the sum of similarities to seed items.
    """

    def __init__(self, recommender: RecipeRecommender) -> None:
        self._rec = recommender
        self._sim: np.ndarray | None = None

    def fit(self, user_train_idx_lists: list[list[int]]) -> None:
        n = len(self._rec.df)
        co = np.zeros((n, n), dtype=np.float64)
        for items in user_train_idx_lists:
            uniq = sorted(set(items))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1 :]:
                    co[a, b] += 1.0
                    co[b, a] += 1.0
        if float(co.sum()) <= 0.0:
            self._sim = np.zeros((n, n), dtype=np.float32)
        else:
            self._sim = cosine_similarity(co).astype(np.float32)
            np.fill_diagonal(self._sim, 0.0)

    def _scores_for_train_idxs(self, train_idxs: list[int]) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("Item-item model is not fitted.")
        n = len(self._rec.df)
        scores = np.zeros(n, dtype=np.float64)
        for t in train_idxs:
            scores += self._sim[t]
        return scores

    def recommend(
        self,
        train_recipe_ids: list[int],
        top_k: int = 10,
        **constraints: Any,
    ) -> pd.DataFrame:
        idxs = [self._rec.id_to_idx[r] for r in train_recipe_ids if r in self._rec.id_to_idx]
        scores = self._scores_for_train_idxs(idxs)
        for i in idxs:
            scores[i] = -np.inf
        scores = self._rec._apply_constraints(scores, **constraints)
        return self._rec._to_result(scores, top_k=top_k)
