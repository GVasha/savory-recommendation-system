from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .core import RecipeRecommender


def _ranking_metrics(hit_positions: list[int], top_k: int) -> dict[str, float]:
    if not hit_positions:
        return {
            f"HitRate@{top_k}": 0.0,
            f"Recall@{top_k}": 0.0,
            f"MRR@{top_k}": 0.0,
            "n_users_evaluated": 0,
        }
    hits = [1 if p >= 0 else 0 for p in hit_positions]
    rrs = [1 / (p + 1) if p >= 0 else 0 for p in hit_positions]
    return {
        f"HitRate@{top_k}": float(np.mean(hits)),
        f"Recall@{top_k}": float(np.mean(hits)),
        f"MRR@{top_k}": float(np.mean(rrs)),
        "n_users_evaluated": int(len(hit_positions)),
    }


def evaluate_with_interactions(
    recommender: RecipeRecommender,
    interactions_path: str | Path,
    top_k: int = 10,
    model: str = "hybrid",
    min_user_interactions: int = 2,
) -> dict[str, float]:
    path = Path(interactions_path)
    if not path.is_file():
        raise FileNotFoundError(f"Interactions file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("Interactions file is empty.")

    interactions = pd.read_csv(path, encoding="utf-8-sig")
    required_cols = {"user_id", "id"}
    if not required_cols.issubset(set(interactions.columns)):
        if {"user_id", "recipe_id"}.issubset(set(interactions.columns)):
            interactions = interactions.rename(columns={"recipe_id": "id"})
        else:
            raise ValueError("Interactions file must contain user_id and id (or recipe_id).")

    if "rating" in interactions.columns:
        interactions = interactions[interactions["rating"].fillna(0) >= 4].copy()
    if "date" in interactions.columns:
        interactions = interactions.sort_values(["user_id", "date"])

    valid_ids = set(recommender.df["id"].tolist())
    interactions = interactions[interactions["id"].isin(valid_ids)].copy()

    user_histories = interactions.groupby("user_id")["id"].apply(list).to_dict()
    users = [u for u, h in user_histories.items() if len(h) >= min_user_interactions]

    hit_positions: list[int] = []
    for u in users:
        hist = user_histories[u]
        train = hist[:-1]
        target = hist[-1]
        recs = recommender.recommend_for_liked(train, top_k=top_k, model=model)
        rec_ids = recs["id"].tolist()
        try:
            pos = rec_ids.index(target)
        except ValueError:
            pos = -1
        hit_positions.append(pos)

    return _ranking_metrics(hit_positions, top_k=top_k)


def evaluate_cold_start_quality(
    recommender: RecipeRecommender,
    top_k: int = 10,
) -> dict[str, float]:
    """
    Quality proxy when no user interactions are available.
    Reports:
    - coverage@k over random seed queries
    - intra-list diversity on TF-IDF space
    """
    ids = recommender.df["id"].tolist()
    if len(ids) < 2:
        return {"coverage@k": 0.0, "avg_diversity@k": 0.0}

    rng = np.random.default_rng(42)
    seeds = rng.choice(ids, size=min(50, len(ids)), replace=False)
    all_recommended_ids: set[int] = set()
    diversities: list[float] = []

    for seed_id in seeds:
        recs = recommender.recommend_similar(int(seed_id), top_k=top_k, model="hybrid")
        rec_ids = recs["id"].tolist()
        all_recommended_ids.update(rec_ids)

        # diversity using TF-IDF similarity among recommended items
        idxs = [recommender.id_to_idx[x] for x in rec_ids if x in recommender.id_to_idx]
        if len(idxs) >= 2:
            sub = cosine_similarity(recommender.X_tfidf[idxs])
            pair_sims = []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pair_sims.append(sub[i, j])
            if pair_sims:
                diversities.append(float(1 - np.mean(pair_sims)))

    coverage = len(all_recommended_ids) / max(len(ids), 1)
    avg_div = float(np.mean(diversities)) if diversities else 0.0
    return {"coverage@k": float(coverage), "avg_diversity@k": avg_div}


def evaluate(
    recommender: RecipeRecommender,
    interactions_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["cold_start"] = evaluate_cold_start_quality(recommender, top_k=top_k)
    if interactions_path:
        try:
            out["with_interactions"] = evaluate_with_interactions(
                recommender,
                interactions_path=interactions_path,
                top_k=top_k,
                model="hybrid",
            )
        except Exception as exc:
            out["with_interactions"] = {"status": f"skipped ({type(exc).__name__}: {exc})"}
    return out
