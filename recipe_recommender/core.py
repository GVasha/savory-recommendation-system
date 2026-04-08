from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .chatbot import QueryPreferences, parse_user_message
from .config import (
    DEFAULT_RECIPES_PATH,
    DEFAULT_W_CATEGORY,
    DEFAULT_W_NUMERIC,
    DEFAULT_W_TEXT,
)
from .preprocessing import (
    build_combined_text,
    ingredient_groups_to_text,
    normalize_bool,
    normalize_text,
    parse_time_to_minutes,
)


def _row_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


class RecipeRecommender:
    def __init__(
        self,
        w_text: float = DEFAULT_W_TEXT,
        w_numeric: float = DEFAULT_W_NUMERIC,
        w_category: float = DEFAULT_W_CATEGORY,
        use_bert: bool = True,
        use_stem: bool = False,
    ) -> None:
        self.w_text = w_text
        self.w_numeric = w_numeric
        self.w_category = w_category
        self.use_bert = use_bert
        self.use_stem = use_stem

        self._sentence_transformer: Any = None

        self.df: pd.DataFrame | None = None
        self.id_to_idx: dict[int, int] = {}
        self.idx_to_id: dict[int, int] = {}

        self.vectorizer: TfidfVectorizer | None = None
        self.X_tfidf = None
        self.X_bert: np.ndarray | None = None
        self.X_text_dense: np.ndarray | None = None
        self.X_numeric: np.ndarray | None = None
        self.X_category: np.ndarray | None = None
        self.X_hybrid: np.ndarray | None = None
        self.bert_source: str = "not_requested"

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for col in [
            "is_vegan",
            "is_vegetarian",
            "is_gluten_free",
            "is_lactose_free",
            "visibility",
        ]:
            if col in out.columns:
                out[col] = out[col].map(normalize_bool)

        if "visibility" in out.columns:
            out = out[out["visibility"] == True].copy()

        if "ingredient_groups" in out.columns:
            out["ingredient_text"] = out["ingredient_groups"].map(ingredient_groups_to_text)
        else:
            out["ingredient_text"] = ""

        if "cooking_time" in out.columns:
            out["cooking_minutes"] = out["cooking_time"].map(parse_time_to_minutes)
        else:
            out["cooking_minutes"] = np.nan

        out["combined_text"] = build_combined_text(out, use_stem=self.use_stem)
        out = out[out["combined_text"].str.len() > 0].copy()
        out = out.drop_duplicates(subset=["id"]).reset_index(drop=True)

        return out

    def _build_bert(self, texts: list[str]) -> np.ndarray | None:
        if not self.use_bert:
            self.bert_source = "disabled"
            self._sentence_transformer = None
            return None
        try:
            from sentence_transformers import SentenceTransformer

            self._sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb = self._sentence_transformer.encode(
                texts,
                show_progress_bar=True,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.bert_source = "sentence-transformers"
            return emb
        except Exception as exc:
            self.bert_source = f"fallback_tfidf ({type(exc).__name__})"
            self._sentence_transformer = None
            return None

    def fit(self, recipes_path: str | Path = DEFAULT_RECIPES_PATH) -> "RecipeRecommender":
        path = Path(recipes_path)
        if not path.is_file():
            raise FileNotFoundError(f"Recipes file not found: {path}")

        raw = pd.read_csv(path, encoding="utf-8-sig")
        self.df = self._prepare_dataframe(raw)
        self.id_to_idx = {int(rid): idx for idx, rid in enumerate(self.df["id"].tolist())}
        self.idx_to_id = {idx: rid for rid, idx in self.id_to_idx.items()}

        texts = self.df["combined_text"].tolist()

        self.vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1)
        self.X_tfidf = self.vectorizer.fit_transform(texts)

        self.X_bert = self._build_bert(texts)
        if self.X_bert is not None:
            self.X_text_dense = self.X_bert.astype(np.float32)
        else:
            self.X_text_dense = _row_normalize(self.X_tfidf.astype(np.float32).toarray())

        numeric_cols = [c for c in ["calories", "carbs", "fats", "proteins", "servings", "cooking_minutes"] if c in self.df.columns]
        if numeric_cols:
            num_df = self.df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            num_df = num_df.fillna(num_df.median(numeric_only=True))
            self.X_numeric = MinMaxScaler().fit_transform(num_df).astype(np.float32)
            self.X_numeric = _row_normalize(self.X_numeric)
        else:
            self.X_numeric = np.zeros((len(self.df), 1), dtype=np.float32)

        if "food_type" in self.df.columns:
            cat_df = pd.get_dummies(self.df["food_type"].fillna("Unknown"), prefix="food")
            self.X_category = cat_df.to_numpy(dtype=np.float32)
            self.X_category = _row_normalize(self.X_category)
        else:
            self.X_category = np.zeros((len(self.df), 1), dtype=np.float32)

        self.X_hybrid = np.hstack(
            [
                self.w_text * _row_normalize(self.X_text_dense),
                self.w_numeric * self.X_numeric,
                self.w_category * self.X_category,
            ]
        )
        self.X_hybrid = _row_normalize(self.X_hybrid)
        return self

    def _query_text_to_hybrid_vector(self, query_text: str) -> np.ndarray:
        self._check_fitted()
        q = normalize_text(query_text, use_stem=self.use_stem)
        if self.X_bert is not None and self._sentence_transformer is not None:
            t = self._sentence_transformer.encode(
                [q],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        else:
            q_vec = self.vectorizer.transform([q])
            t = _row_normalize(q_vec.astype(np.float32).toarray())

        n_num = self.X_numeric.shape[1]
        n_cat = self.X_category.shape[1]
        z_num = np.zeros((1, n_num), dtype=np.float32)
        z_cat = np.zeros((1, n_cat), dtype=np.float32)
        part1 = self.w_text * _row_normalize(t)
        part2 = self.w_numeric * z_num
        part3 = self.w_category * z_cat
        q_h = np.hstack([part1, part2, part3])
        q_h = _row_normalize(q_h)
        return q_h.ravel().astype(np.float32)

    def _check_fitted(self) -> None:
        if self.df is None or self.vectorizer is None or self.X_hybrid is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def _score_by_model(self, model: str, query_text: str | None = None, idx: int | None = None) -> np.ndarray:
        self._check_fitted()
        model = model.lower()

        if idx is not None:
            if model == "tfidf":
                return cosine_similarity(self.X_tfidf[idx], self.X_tfidf).ravel()
            if model == "bert":
                base = self.X_bert if self.X_bert is not None else self.X_text_dense
                return (base @ base[idx]).ravel()
            if model == "hybrid":
                return (self.X_hybrid @ self.X_hybrid[idx]).ravel()
            raise ValueError(f"Unsupported model: {model}")

        if query_text is None:
            raise ValueError("query_text is required for query scoring.")
        q = normalize_text(query_text, use_stem=self.use_stem)

        if model == "tfidf":
            q_vec = self.vectorizer.transform([q])
            return cosine_similarity(q_vec, self.X_tfidf).ravel()
        if model == "bert":
            if self.X_bert is not None and self._sentence_transformer is not None:
                q_vec = self._sentence_transformer.encode(
                    [q], convert_to_numpy=True, normalize_embeddings=True
                )
                return (q_vec @ self.X_bert.T).ravel()
            q_vec = self.vectorizer.transform([q])
            return cosine_similarity(q_vec, self.X_tfidf).ravel()
        if model == "hybrid":
            q_h = self._query_text_to_hybrid_vector(query_text)
            return (self.X_hybrid @ q_h).ravel()

        raise ValueError(f"Unsupported model: {model}")

    def _apply_constraints(
        self,
        scores: np.ndarray,
        max_minutes: int | None = None,
        food_type: str | None = None,
        is_vegan: bool | None = None,
        is_vegetarian: bool | None = None,
        is_gluten_free: bool | None = None,
        is_lactose_free: bool | None = None,
        include_terms: list[str] | None = None,
        exclude_terms: list[str] | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        mask = np.ones(len(self.df), dtype=bool)

        if max_minutes is not None and "cooking_minutes" in self.df.columns:
            minutes = pd.to_numeric(self.df["cooking_minutes"], errors="coerce")
            mask &= (minutes <= max_minutes) | minutes.isna()

        if food_type and "food_type" in self.df.columns:
            mask &= self.df["food_type"].fillna("").str.lower().eq(food_type.lower())

        for col, val in [
            ("is_vegan", is_vegan),
            ("is_vegetarian", is_vegetarian),
            ("is_gluten_free", is_gluten_free),
            ("is_lactose_free", is_lactose_free),
        ]:
            if val is not None and col in self.df.columns:
                mask &= self.df[col].fillna(False).astype(bool).eq(val)

        text_field = self.df["combined_text"].fillna("")
        if include_terms:
            for t in include_terms:
                token = normalize_text(t, use_stem=self.use_stem)
                if token:
                    mask &= text_field.str.contains(token, regex=False)
        if exclude_terms:
            for t in exclude_terms:
                token = normalize_text(t, use_stem=self.use_stem)
                if token:
                    mask &= ~text_field.str.contains(token, regex=False)

        out = np.where(mask, scores, -np.inf)
        return out

    def _to_result(self, scores: np.ndarray, top_k: int) -> pd.DataFrame:
        self._check_fitted()
        top_idx = np.argsort(scores)[::-1]
        top_idx = [i for i in top_idx if np.isfinite(scores[i])][:top_k]
        out = self.df.iloc[top_idx].copy()
        out["score"] = scores[top_idx]
        cols = [c for c in ["id", "name", "food_type", "cooking_time", "calories", "proteins", "score"] if c in out.columns]
        return out[cols]

    def recommend_similar(self, recipe_id: int, top_k: int = 10, model: str = "hybrid", **constraints: Any) -> pd.DataFrame:
        self._check_fitted()
        rid = int(recipe_id)
        if rid not in self.id_to_idx:
            raise ValueError(f"Recipe ID {rid} not found in dataset.")
        idx = self.id_to_idx[rid]
        scores = self._score_by_model(model=model, idx=idx)
        scores[idx] = -np.inf
        scores = self._apply_constraints(scores, **constraints)
        return self._to_result(scores, top_k=top_k)

    def recommend_for_query(self, query: str, top_k: int = 10, model: str = "hybrid", **constraints: Any) -> pd.DataFrame:
        self._check_fitted()
        scores = self._score_by_model(model=model, query_text=query)
        scores = self._apply_constraints(scores, **constraints)
        return self._to_result(scores, top_k=top_k)

    def recommend_for_liked(self, liked_recipe_ids: list[int], top_k: int = 10, model: str = "hybrid", **constraints: Any) -> pd.DataFrame:
        self._check_fitted()
        idxs = [self.id_to_idx[int(x)] for x in liked_recipe_ids if int(x) in self.id_to_idx]
        if not idxs:
            return pd.DataFrame(columns=["id", "name", "score"])

        if model.lower() == "hybrid":
            profile = self.X_hybrid[idxs].mean(axis=0)
            scores = (self.X_hybrid @ profile).ravel()
        elif model.lower() == "bert":
            base = self.X_bert if self.X_bert is not None else self.X_text_dense
            profile = base[idxs].mean(axis=0)
            scores = (base @ profile).ravel()
        else:
            profile = np.asarray(self.X_tfidf[idxs].mean(axis=0))
            scores = cosine_similarity(profile, self.X_tfidf).ravel()

        for i in idxs:
            scores[i] = -np.inf
        scores = self._apply_constraints(scores, **constraints)
        return self._to_result(scores, top_k=top_k)

    def recommend_from_message(self, message: str, top_k: int = 10, model: str = "hybrid") -> tuple[pd.DataFrame, dict[str, Any]]:
        prefs = parse_user_message(message)
        pref_dict = asdict(prefs)
        query_text = pref_dict.pop("query_text")
        recs = self.recommend_for_query(query_text, top_k=top_k, model=model, **pref_dict)
        return recs, pref_dict

    def recommend_random(
        self,
        top_k: int = 10,
        random_state: int = 42,
        exclude_recipe_ids: list[int] | None = None,
        **constraints: Any,
    ) -> pd.DataFrame:
        self._check_fitted()
        rng = np.random.default_rng(random_state)
        scores = rng.random(len(self.df), dtype=np.float64)
        scores = self._apply_constraints(scores, **constraints)
        if exclude_recipe_ids:
            for rid in exclude_recipe_ids:
                if int(rid) in self.id_to_idx:
                    scores[self.id_to_idx[int(rid)]] = -np.inf
        return self._to_result(scores, top_k=top_k)

    def recommend_switching(
        self,
        user_recipe_history: list[int],
        user_id: int | None = None,
        svd_model: Any = None,
        min_cf_interactions: int = 3,
        top_k: int = 10,
        **constraints: Any,
    ) -> tuple[pd.DataFrame, str]:
        """
        Switching hybrid: route to SVD-based CF when the user has sufficient catalog
        interactions and a fitted SVD model is available; otherwise fall back to
        content-based hybrid (liked-profile or popular for cold-start users).

        Returns
        -------
        (DataFrame of recommendations, strategy_name)
        """
        self._check_fitted()
        catalog_history = [int(r) for r in user_recipe_history if int(r) in self.id_to_idx]

        can_use_cf = (
            svd_model is not None
            and user_id is not None
            and len(catalog_history) >= min_cf_interactions
            and int(user_id) in svd_model.user_map
        )

        if can_use_cf:
            rec_ids = svd_model.recommend(
                int(user_id), top_k=top_k * 2, exclude_ids=catalog_history
            )
            rec_ids = [r for r in rec_ids if r in self.id_to_idx][:top_k]
            if rec_ids:
                scores = np.full(len(self.df), -np.inf, dtype=np.float64)
                u_idx = svd_model.user_map[int(user_id)]
                pred = svd_model._pred_vector(u_idx)
                for rid in rec_ids:
                    i_svd = svd_model.item_map.get(int(rid))
                    df_idx = self.id_to_idx[int(rid)]
                    scores[df_idx] = float(pred[i_svd]) if i_svd is not None else 0.0
                scores = self._apply_constraints(scores, **constraints)
                return self._to_result(scores, top_k=top_k), "svd_cf"

        if catalog_history:
            return (
                self.recommend_for_liked(
                    catalog_history, top_k=top_k, model="hybrid", **constraints
                ),
                "content_hybrid",
            )
        return self.recommend_popular(top_k=top_k, **constraints), "popular_cold_start"

    def recommend_popular(
        self,
        top_k: int = 10,
        popularity_by_id: dict[int, float] | None = None,
        exclude_recipe_ids: list[int] | None = None,
        **constraints: Any,
    ) -> pd.DataFrame:
        """
        Non-personalized popularity baseline. If popularity_by_id is None, uses a
        uniform prior over the catalog (still respects hard constraints).
        """
        self._check_fitted()
        if popularity_by_id is None:
            scores = np.ones(len(self.df), dtype=np.float64)
        else:
            scores = np.array(
                [float(popularity_by_id.get(int(rid), 0.0)) for rid in self.df["id"].tolist()],
                dtype=np.float64,
            )
        scores = self._apply_constraints(scores, **constraints)
        if exclude_recipe_ids:
            for rid in exclude_recipe_ids:
                if int(rid) in self.id_to_idx:
                    scores[self.id_to_idx[int(rid)]] = -np.inf
        return self._to_result(scores, top_k=top_k)
