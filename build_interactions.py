"""
Parse PP_users.csv and extract user-item interactions that overlap with the recipe catalog.

Outputs:
  data/interactions_catalog.csv  - all catalog interactions (user_id, id, rating)
  data/interactions_train.csv    - first 80% of each user's history (time-based proxy)
  data/interactions_test.csv     - last 20% of each user's history

Usage:
  python build_interactions.py
  python build_interactions.py --min-rating 3.0 --test-ratio 0.2 --min-interactions 3
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def _parse_list_col(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    return []


def parse_pp_users(
    pp_users_path: Path,
    catalog_ids: set[int],
    min_rating: float = 1.0,
) -> pd.DataFrame:
    df = pd.read_csv(pp_users_path, encoding="utf-8-sig")
    rows = []
    for row in df.itertuples(index=False):
        user_id = int(row.u)
        items = _parse_list_col(row.items)
        ratings = _parse_list_col(row.ratings)
        for item_id, rating in zip(items, ratings):
            item_id = int(item_id)
            if item_id == 0:
                continue
            if item_id in catalog_ids and float(rating) >= min_rating:
                rows.append({"user_id": user_id, "id": item_id, "rating": float(rating)})
    return pd.DataFrame(rows, columns=["user_id", "id", "rating"])


def time_based_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by position in each user's interaction list (proxy for time — earlier = older).
    Users with fewer than min_interactions interactions go entirely to train.
    """
    train_parts, test_parts = [], []
    for _, group in interactions.groupby("user_id", sort=False):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < min_interactions:
            train_parts.append(group)
        else:
            cut = max(1, int(n * (1 - test_ratio)))
            train_parts.append(group.iloc[:cut])
            test_parts.append(group.iloc[cut:])

    empty = pd.DataFrame(columns=["user_id", "id", "rating"])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else empty
    test = pd.concat(test_parts, ignore_index=True) if test_parts else empty
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build catalog interaction files from PP_users.csv"
    )
    parser.add_argument("--pp-users", type=Path, default=DATA_DIR / "PP_users.csv")
    parser.add_argument("--recipes", type=Path, default=DATA_DIR / "recipes_visible_only.csv")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--min-rating", type=float, default=1.0, help="Minimum rating to keep.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=3,
        help="Users with fewer interactions go entirely to train.",
    )
    args = parser.parse_args()

    catalog = pd.read_csv(args.recipes, encoding="utf-8-sig")
    catalog_ids: set[int] = set(catalog["id"].astype(int).tolist())
    print(f"Catalog: {len(catalog_ids)} recipes (IDs {min(catalog_ids)}–{max(catalog_ids)})")

    print("Parsing PP_users.csv (this may take ~30s)…")
    interactions = parse_pp_users(args.pp_users, catalog_ids, min_rating=args.min_rating)

    if interactions.empty:
        print("No overlapping interactions found. Check that recipe IDs match.")
        return

    print(f"  Total interactions : {len(interactions):,}")
    print(f"  Unique users       : {interactions['user_id'].nunique():,}")
    print(f"  Catalog recipes covered: {interactions['id'].nunique()} / {len(catalog_ids)}")
    print(f"  Rating distribution:\n{interactions['rating'].value_counts().sort_index().to_string()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.out_dir / "interactions_catalog.csv"
    interactions.to_csv(all_path, index=False)
    print(f"\nSaved: {all_path}")

    train, test = time_based_split(
        interactions, test_ratio=args.test_ratio, min_interactions=args.min_interactions
    )
    train_path = args.out_dir / "interactions_train.csv"
    test_path = args.out_dir / "interactions_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Saved: {train_path}  ({len(train):,} rows, {train['user_id'].nunique():,} users)")
    print(f"Saved: {test_path}   ({len(test):,} rows,  {test['user_id'].nunique():,} users)")


if __name__ == "__main__":
    main()
