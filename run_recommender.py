from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recipe_recommender.config import (
    DATA_DIR,
    DEFAULT_INTERACTIONS_PATH,
    DEFAULT_RECIPES_PATH,
)
from recipe_recommender.core import RecipeRecommender
from recipe_recommender.evaluation import evaluate


def _parse_liked_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(
            text.encode("utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            )
        )


def _load_svd(interactions_path: Path, recipes_path: Path, n_factors: int = 20):
    """Fit and return an SVDRecommender, or None if interactions are unavailable."""
    from recipe_recommender.svd_model import SVDRecommender
    import pandas as pd

    if not interactions_path.is_file() or interactions_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(interactions_path, encoding="utf-8-sig")
        if not {"user_id", "id", "rating"}.issubset(df.columns) or df.empty:
            return None
        svd = SVDRecommender(n_factors=n_factors)
        svd.fit(df)
        return svd
    except Exception as exc:
        print(f"[SVD] Could not fit model: {exc}", file=sys.stderr)
        return None


def build_parser() -> argparse.ArgumentParser:
    root_topk = argparse.ArgumentParser(add_help=False)
    root_topk.add_argument("--top-k", type=int, default=10, help="Number of results.")

    sub_topk = argparse.ArgumentParser(add_help=False)
    sub_topk.add_argument(
        "--top-k",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of results (can also be set before the subcommand).",
    )

    parser = argparse.ArgumentParser(
        description="Recipe recommender — query, evaluate, and demo all strategies.",
        parents=[root_topk],
    )
    parser.add_argument("--recipes-path", type=Path, default=DEFAULT_RECIPES_PATH)
    parser.add_argument("--interactions-path", type=Path, default=DEFAULT_INTERACTIONS_PATH)
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DATA_DIR / "interactions_test.csv",
        help="Held-out ratings for RMSE/MAE evaluation.",
    )
    parser.add_argument("--model", choices=["hybrid", "tfidf", "bert"], default="hybrid")
    parser.add_argument("--no-bert", action="store_true")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--svd-factors", type=int, default=20, help="SVD latent factors.")

    # Filters shared across query/similar/liked/message
    parser.add_argument("--max-minutes", type=int, default=None)
    parser.add_argument("--food-type", type=str, default=None)
    parser.add_argument("--vegan", action="store_true")
    parser.add_argument("--vegetarian", action="store_true")
    parser.add_argument("--gluten-free", action="store_true")
    parser.add_argument("--lactose-free", action="store_true")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- query ---
    q = sub.add_parser("query", parents=[sub_topk], help="Recommend from free-text query.")
    q.add_argument("--text", required=True)

    # --- similar ---
    s = sub.add_parser("similar", parents=[sub_topk], help="Recommend similar to one recipe.")
    s.add_argument("--recipe-id", required=True, type=int)

    # --- liked ---
    l = sub.add_parser("liked", parents=[sub_topk], help="Recommend from liked recipe IDs.")
    l.add_argument("--ids", required=True, help="Comma-separated recipe IDs.")
    l.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Known user ID for SVD-based personalisation (optional).",
    )
    l.add_argument(
        "--switching",
        action="store_true",
        help="Use switching hybrid: SVD when available, else content-based.",
    )

    # --- message ---
    m = sub.add_parser("message", parents=[sub_topk], help="Chatbot-style recommendation.")
    m.add_argument("--text", required=True)

    # --- eval ---
    e = sub.add_parser("eval", parents=[sub_topk], help="Run evaluation suite.")
    e.add_argument(
        "--compare",
        action="store_true",
        help="Leave-one-out comparison: hybrid / tfidf / random / popular / item-item CF / SVD.",
    )
    e.add_argument("--no-svd", action="store_true", help="Skip SVD in the comparison.")

    # --- bandit-demo ---
    bd = sub.add_parser(
        "bandit-demo",
        help="Simulate epsilon-greedy bandit over random / popular / hybrid arms.",
    )
    bd.add_argument("--rounds", type=int, default=200)
    bd.add_argument("--epsilon", type=float, default=0.1)
    bd.add_argument("--seed", type=int, default=42)

    # --- build-interactions ---
    bi = sub.add_parser(
        "build-interactions",
        help="Parse PP_users.csv to create interaction files in data/.",
    )
    bi.add_argument("--pp-users", type=Path, default=DATA_DIR / "PP_users.csv")
    bi.add_argument("--min-rating", type=float, default=1.0)
    bi.add_argument("--test-ratio", type=float, default=0.2)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # build-interactions does not need the recommender fitted
    if args.command == "build-interactions":
        from build_interactions import main as bi_main
        import sys as _sys

        _sys.argv = [
            "build_interactions.py",
            "--pp-users", str(args.pp_users),
            "--recipes", str(args.recipes_path),
            "--min-rating", str(args.min_rating),
            "--test-ratio", str(args.test_ratio),
        ]
        bi_main()
        return

    include_terms = [x.strip() for x in args.include.split(",") if x.strip()]
    exclude_terms = [x.strip() for x in args.exclude.split(",") if x.strip()]
    filters = {
        "max_minutes": args.max_minutes,
        "food_type": args.food_type,
        "is_vegan": True if args.vegan else None,
        "is_vegetarian": True if args.vegetarian else None,
        "is_gluten_free": True if args.gluten_free else None,
        "is_lactose_free": True if args.lactose_free else None,
        "include_terms": include_terms or None,
        "exclude_terms": exclude_terms or None,
    }

    rec = RecipeRecommender(use_bert=not args.no_bert, use_stem=args.stem)
    rec.fit(args.recipes_path)
    print(f"Loaded {len(rec.df)} visible recipes  |  BERT: {rec.bert_source}")

    if args.command == "query":
        out = rec.recommend_for_query(args.text, top_k=args.top_k, model=args.model, **filters)
        _safe_print(out.to_string(index=False))
        return

    if args.command == "similar":
        out = rec.recommend_similar(
            args.recipe_id, top_k=args.top_k, model=args.model, **filters
        )
        _safe_print(out.to_string(index=False))
        return

    if args.command == "liked":
        liked_ids = _parse_liked_ids(args.ids)
        if getattr(args, "switching", False):
            svd = _load_svd(args.interactions_path, args.recipes_path, args.svd_factors)
            out, strategy = rec.recommend_switching(
                liked_ids,
                user_id=getattr(args, "user_id", None),
                svd_model=svd,
                top_k=args.top_k,
                **filters,
            )
            print(f"Strategy used: {strategy}")
        else:
            out = rec.recommend_for_liked(
                liked_ids, top_k=args.top_k, model=args.model, **filters
            )
        _safe_print(out.to_string(index=False))
        return

    if args.command == "message":
        out, parsed = rec.recommend_from_message(
            args.text, top_k=args.top_k, model=args.model
        )
        print("Parsed preferences:", parsed)
        _safe_print(out.to_string(index=False))
        return

    if args.command == "bandit-demo":
        from recipe_recommender.bandit import simulate_bandit
        import json

        print(f"Running bandit simulation: {args.rounds} rounds, eps={args.epsilon} ...")
        result = simulate_bandit(
            rec, n_rounds=args.rounds, epsilon=args.epsilon, random_seed=args.seed
        )
        summary = {k: v for k, v in result.items() if k != "history"}
        print(json.dumps(summary, indent=2))

        # Print last 5 rounds of history
        print("\nLast 5 rounds:")
        h = result["history"]
        for i in range(-5, 0):
            print(
                f"  Round {h['round'][i]:>3}  arm={h['arm_name'][i]:<8}"
                f"  reward={h['reward'][i]:.4f}  cumulative={h['cumulative_reward'][i]:.4f}"
            )
        return

    if args.command == "eval":
        compare = getattr(args, "compare", False)
        use_svd = not getattr(args, "no_svd", False)
        svd = None
        if use_svd:
            svd = _load_svd(args.interactions_path, args.recipes_path, args.svd_factors)
            if svd:
                print(
                    f"SVD fitted: {svd.n_users} users, {svd.n_items} items, "
                    f"{svd._n_factors_actual} factors"
                )
            else:
                print("SVD: skipped (no interaction data)")

        import json
        results = evaluate(
            rec,
            interactions_path=args.interactions_path if args.interactions_path.is_file() else None,
            test_path=args.test_path if args.test_path.is_file() else None,
            top_k=args.top_k,
            compare_baselines=compare,
            svd_model=svd,
        )
        print(json.dumps(results, indent=2, default=str))
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
