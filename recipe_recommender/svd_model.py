"""
Model-based collaborative filtering via truncated SVD on the user-item rating matrix.

Handles explicit ratings (1-5). Unobserved entries are treated as 0 after per-user
mean-centering so the SVD focuses on deviations from each user's average taste.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

if TYPE_CHECKING:
    from .core import RecipeRecommender


class SVDRecommender:
    """
    Truncated SVD collaborative filter.

    Typical usage
    -------------
    svd = SVDRecommender(n_factors=20)
    svd.fit(train_df)                               # train_df: user_id, id, rating
    svd.recommend_as_dataframe(user_id=7, rec=rec)  # rec = fitted RecipeRecommender
    svd.rmse_mae(test_df)                           # regression-style evaluation
    """

    def __init__(self, n_factors: int = 20) -> None:
        self.n_factors = n_factors
        self.user_map: dict[int, int] = {}
        self.item_map: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self.user_means: np.ndarray | None = None
        self._R_hat: np.ndarray | None = None
        self.global_mean: float = 0.0
        self._n_factors_actual: int = 0

    def fit(self, interactions: pd.DataFrame) -> "SVDRecommender":
        """
        Fit the model from a DataFrame with columns: user_id, id, rating.
        """
        interactions = interactions.copy()
        interactions["user_id"] = interactions["user_id"].astype(int)
        interactions["id"] = interactions["id"].astype(int)
        interactions["rating"] = interactions["rating"].astype(float)

        users = sorted(interactions["user_id"].unique())
        items = sorted(interactions["id"].unique())
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {r: i for i, r in enumerate(items)}
        self._idx_to_item = {i: r for r, i in self.item_map.items()}
        self.global_mean = float(interactions["rating"].mean())

        n_u, n_i = len(users), len(items)
        R = np.zeros((n_u, n_i), dtype=np.float32)
        for row in interactions.itertuples(index=False):
            R[self.user_map[row.user_id], self.item_map[row.id]] = row.rating

        # Per-user mean over observed entries
        observed = R != 0
        row_counts = observed.sum(axis=1).clip(min=1)
        self.user_means = (R.sum(axis=1) / row_counts).astype(np.float32)

        R_c = R.copy()
        for u in range(n_u):
            R_c[u, observed[u]] -= self.user_means[u]

        k = min(self.n_factors, min(n_u, n_i) - 1)
        self._n_factors_actual = k

        U, s, Vt = svds(csr_matrix(R_c.astype(np.float64)), k=k)
        # svds returns ascending order — reverse to descending
        order = np.argsort(s)[::-1]
        U, s, Vt = U[:, order], s[order], Vt[order, :]

        self._R_hat = ((U * s) @ Vt).astype(np.float32)
        return self

    def _check_fitted(self) -> None:
        if self._R_hat is None:
            raise RuntimeError("SVDRecommender is not fitted. Call fit() first.")

    def _pred_vector(self, user_idx: int) -> np.ndarray:
        return np.clip(
            self.user_means[user_idx] + self._R_hat[user_idx],
            1.0,
            5.0,
        )

    def predict_rating(self, user_id: int, recipe_id: int) -> float | None:
        """Predict rating for a (user, item) pair. Returns None if unknown."""
        self._check_fitted()
        u = self.user_map.get(int(user_id))
        i = self.item_map.get(int(recipe_id))
        if u is None or i is None:
            return None
        return float(self._pred_vector(u)[i])

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_ids: list[int] | None = None,
    ) -> list[int]:
        """Return top-k recipe IDs ordered by predicted rating."""
        self._check_fitted()
        u = self.user_map.get(int(user_id))
        if u is None:
            return []
        pred = self._pred_vector(u)
        exclude_set = {int(x) for x in (exclude_ids or [])}
        result: list[int] = []
        for i in np.argsort(pred)[::-1]:
            rid = self._idx_to_item[int(i)]
            if rid not in exclude_set:
                result.append(rid)
            if len(result) >= top_k:
                break
        return result

    def recommend_as_dataframe(
        self,
        user_id: int,
        rec: "RecipeRecommender",
        top_k: int = 10,
        exclude_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Return a formatted result DataFrame matching RecipeRecommender output style."""
        self._check_fitted()
        u = self.user_map.get(int(user_id))
        if u is None:
            return pd.DataFrame(columns=["id", "name", "score"])

        pred = self._pred_vector(u)
        exclude_set = {int(x) for x in (exclude_ids or [])}

        scores = np.full(len(rec.df), -np.inf, dtype=np.float32)
        for i, rating in enumerate(pred):
            rid = self._idx_to_item[i]
            if rid in rec.id_to_idx and rid not in exclude_set:
                scores[rec.id_to_idx[rid]] = rating
        return rec._to_result(scores, top_k=top_k)

    def rmse_mae(self, test_df: pd.DataFrame) -> dict[str, Any]:
        """
        Evaluate rating prediction quality on a held-out test set.
        Regression metrics: RMSE and MAE.
        """
        self._check_fitted()
        preds, actuals = [], []
        n_cold = 0
        for row in test_df.itertuples(index=False):
            p = self.predict_rating(int(row.user_id), int(row.id))
            if p is None:
                n_cold += 1
            else:
                preds.append(p)
                actuals.append(float(row.rating))

        if not preds:
            return {"RMSE": None, "MAE": None, "n_predictions": 0, "n_cold_start": n_cold}

        p_arr = np.array(preds)
        a_arr = np.array(actuals)
        return {
            "RMSE": float(np.sqrt(np.mean((p_arr - a_arr) ** 2))),
            "MAE": float(np.mean(np.abs(p_arr - a_arr))),
            "n_predictions": len(preds),
            "n_cold_start": n_cold,
        }

    @property
    def n_users(self) -> int:
        return len(self.user_map)

    @property
    def n_items(self) -> int:
        return len(self.item_map)
