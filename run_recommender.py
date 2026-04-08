from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recipe_recommender.config import DEFAULT_INTERACTIONS_PATH, DEFAULT_RECIPES_PATH
from recipe_recommender.core import RecipeRecommender
from recipe_recommender.evaluation import evaluate


def _parse_liked_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    # --top-k on the root parser (default 10). On subparsers use SUPPRESS so a value given only
    # before the subcommand is not overwritten by the subparser default.
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
        description="Recipe recommender for recipes_visible_only.csv",
        parents=[root_topk],
    )
    parser.add_argument("--recipes-path", type=Path, default=DEFAULT_RECIPES_PATH, help="Path to recipes CSV.")
    parser.add_argument(
        "--interactions-path",
        type=Path,
        default=DEFAULT_INTERACTIONS_PATH,
        help="Path to interactions CSV (optional, may be empty).",
    )
    parser.add_argument("--model", choices=["hybrid", "tfidf", "bert"], default="hybrid")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERT and use TF-IDF only.")
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Use Porter stemming for catalog text if nltk is installed (optional).",
    )

    # filters
    parser.add_argument("--max-minutes", type=int, default=None)
    parser.add_argument("--food-type", type=str, default=None)
    parser.add_argument("--vegan", action="store_true")
    parser.add_argument("--vegetarian", action="store_true")
    parser.add_argument("--gluten-free", action="store_true")
    parser.add_argument("--lactose-free", action="store_true")
    parser.add_argument("--include", type=str, default="", help="Comma-separated include terms.")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated exclude terms.")

    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", parents=[sub_topk], help="Recommend from free-text query.")
    q.add_argument("--text", required=True, type=str)

    s = sub.add_parser("similar", parents=[sub_topk], help="Recommend similar recipes to one recipe id.")
    s.add_argument("--recipe-id", required=True, type=int)

    l = sub.add_parser("liked", parents=[sub_topk], help="Recommend from list of liked recipe ids.")
    l.add_argument("--ids", required=True, type=str, help="Comma-separated recipe IDs.")

    m = sub.add_parser("message", parents=[sub_topk], help="Chatbot-style recommendation from a user message.")
    m.add_argument("--text", required=True, type=str)

    e = sub.add_parser("eval", parents=[sub_topk], help="Run evaluation suite.")
    e.add_argument(
        "--compare",
        action="store_true",
        help="When interactions file is non-empty, run leave-one-out comparison "
        "(hybrid, tf-idf, random, popularity, item-item CF).",
    )
    return parser


def _safe_print(text: str) -> None:
    """Print with UTF-8 fallback for Windows terminals that don't support all Unicode."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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

    recommender = RecipeRecommender(use_bert=not args.no_bert, use_stem=args.stem)
    recommender.fit(args.recipes_path)
    print(f"Loaded {len(recommender.df)} visible recipes")
    print(f"BERT source: {recommender.bert_source}")

    if args.command == "query":
        out = recommender.recommend_for_query(args.text, top_k=args.top_k, model=args.model, **filters)
        _safe_print(out.to_string(index=False))
        return

    if args.command == "similar":
        out = recommender.recommend_similar(args.recipe_id, top_k=args.top_k, model=args.model, **filters)
        _safe_print(out.to_string(index=False))
        return

    if args.command == "liked":
        liked_ids = _parse_liked_ids(args.ids)
        out = recommender.recommend_for_liked(liked_ids, top_k=args.top_k, model=args.model, **filters)
        _safe_print(out.to_string(index=False))
        return

    if args.command == "message":
        out, parsed = recommender.recommend_from_message(args.text, top_k=args.top_k, model=args.model)
        print("Parsed preferences:", parsed)
        _safe_print(out.to_string(index=False))
        return

    if args.command == "eval":
        compare = getattr(args, "compare", False)
        results = evaluate(
            recommender,
            interactions_path=args.interactions_path if args.interactions_path.is_file() else None,
            top_k=args.top_k,
            compare_baselines=compare,
        )
        print(results)
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
