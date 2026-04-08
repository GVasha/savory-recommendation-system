from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .collaborative import ItemItemCollaborativeFiltering
from .core import RecipeRecommender


def _ndcg_binary(hit_pos: int, top_k: int) -> float:
    """One relevant item at unknown rank; DCG uses relevance 1 at hit_pos (0-based)."""
    if hit_pos < 0 or hit_pos >= top_k:
        return 0.0
    dcg = 1.0 / np.log2(hit_pos + 2)
    idcg = 1.0 / np.log2(2)
    return float(dcg / idcg)


def _ranking_metrics(hit_positions: list[int], top_k: int) -> dict[str, float]:
    if not hit_positions:
        return {
            f"HitRate@{top_k}": 0.0,
            f"Recall@{top_k}": 0.0,
            f"MRR@{top_k}": 0.0,
            f"NDCG@{top_k}": 0.0,
            "n_users_evaluated": 0,
        }
    hits = [1 if p >= 0 else 0 for p in hit_positions]
    rrs = [1 / (p + 1) if p >= 0 else 0 for p in hit_positions]
    ndcgs = [_ndcg_binary(p, top_k) for p in hit_positions]
    return {
        f"HitRate@{top_k}": float(np.mean(hits)),
        f"Recall@{top_k}": float(np.mean(hits)),
        f"MRR@{top_k}": float(np.mean(rrs)),
        f"NDCG@{top_k}": float(np.mean(ndcgs)),
        "n_users_evaluated": int(len(hit_positions)),
    }


def _prepare_interactions(
    path: Path,
    recommender: RecipeRecommender,
    min_user_interactions: int,
) -> tuple[dict[Any, list[Any]], list[Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Interactions file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("Interactions file is empty.")

    interactions = pd.read_csv(path, encoding="utf-8-sig")
    if not {"user_id", "id"}.issubset(interactions.columns):
        if {"user_id", "recipe_id"}.issubset(interactions.columns):
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
    return user_histories, users


def _hit_position_from_rec_ids(rec_ids: list[Any], target: int) -> int:
    try:
        return rec_ids.index(target)
    except ValueError:
        return -1


def _eval_strategy(
    users: list[Any],
    user_histories: dict[Any, list[Any]],
    top_k: int,
    recommend_fn: Callable[[int, list[int], int], list[int]],
) -> dict[str, float]:
    positions: list[int] = []
    for i, u in enumerate(users):
        hist = user_histories[u]
        train = [int(x) for x in hist[:-1]]
        target = int(hist[-1])
        rec_ids = recommend_fn(i, train, target)
        positions.append(_hit_position_from_rec_ids(rec_ids, target))
    return _ranking_metrics(positions, top_k=top_k)


def evaluate_with_interactions(
    recommender: RecipeRecommender,
    interactions_path: str | Path,
    top_k: int = 10,
    model: str = "hybrid",
    min_user_interactions: int = 2,
) -> dict[str, float]:
    user_histories, users = _prepare_interactions(
        Path(interactions_path), recommender, min_user_interactions
    )

    def recommend_fn(_i: int, train: list[int], _target: int) -> list[int]:
        recs = recommender.recommend_for_liked(train, top_k=top_k, model=model)
        return recs["id"].tolist()

    return _eval_strategy(users, user_histories, top_k, recommend_fn)


def evaluate_leave_one_out_comparison(
    recommender: RecipeRecommender,
    interactions_path: str | Path,
    top_k: int = 10,
    min_user_interactions: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Leave-last-item-out for each user: compare hybrid, TF-IDF, random, popularity,
    and memory-based item-item collaborative filtering trained on training interactions.
    Popularity counts use only each user's train slice aggregated across users.
    """
    path = Path(interactions_path)
    user_histories, users = _prepare_interactions(path, recommender, min_user_interactions)
    if not users:
        return {"status": "no users with sufficient interactions"}

    users = sorted(users, key=lambda x: str(x))
    popularity_train = Counter()
    train_idx_lists: list[list[int]] = []
    for u in users:
        hist = user_histories[u]
        train_ids = [int(x) for x in hist[:-1]]
        for rid in train_ids:
            popularity_train[rid] += 1
        train_idx_lists.append(
            [recommender.id_to_idx[r] for r in train_ids if r in recommender.id_to_idx]
        )

    popularity_dict: dict[int, float] = {int(k): float(v) for k, v in popularity_train.items()}

    ii_cf = ItemItemCollaborativeFiltering(recommender)
    ii_cf.fit(train_idx_lists)

    out: dict[str, Any] = {}

    out["hybrid"] = _eval_strategy(
        users,
        user_histories,
        top_k,
        lambda _i, train, _t: recommender.recommend_for_liked(train, top_k=top_k, model="hybrid")[
            "id"
        ].tolist(),
    )

    out["tfidf"] = _eval_strategy(
        users,
        user_histories,
        top_k,
        lambda _i, train, _t: recommender.recommend_for_liked(train, top_k=top_k, model="tfidf")[
            "id"
        ].tolist(),
    )

    out["random"] = _eval_strategy(
        users,
        user_histories,
        top_k,
        lambda i, train, _t: recommender.recommend_random(
            top_k=top_k,
            random_state=random_seed + i,
            exclude_recipe_ids=train,
        )["id"].tolist(),
    )

    out["popular"] = _eval_strategy(
        users,
        user_histories,
        top_k,
        lambda _i, train, _t: recommender.recommend_popular(
            top_k=top_k,
            popularity_by_id=popularity_dict,
            exclude_recipe_ids=train,
        )["id"].tolist(),
    )

    out["item_item_cf"] = _eval_strategy(
        users,
        user_histories,
        top_k,
        lambda _i, train, _t: ii_cf.recommend(train, top_k=top_k)["id"].tolist(),
    )

    out["n_users_evaluated"] = len(users)
    return out


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
    compare_baselines: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["cold_start"] = evaluate_cold_start_quality(recommender, top_k=top_k)
    if interactions_path:
        path = Path(interactions_path)
        if path.is_file() and path.stat().st_size > 0:
            if compare_baselines:
                out["leave_one_out_comparison"] = evaluate_leave_one_out_comparison(
                    recommender,
                    interactions_path=path,
                    top_k=top_k,
                )
            else:
                try:
                    out["with_interactions"] = evaluate_with_interactions(
                        recommender,
                        interactions_path=path,
                        top_k=top_k,
                        model="hybrid",
                    )
                except Exception as exc:
                    out["with_interactions"] = {"status": f"skipped ({type(exc).__name__}: {exc})"}
        else:
            out["with_interactions"] = {"status": "skipped (missing or empty interactions file)"}
    return out
