from __future__ import annotations

import argparse
from pathlib import Path

from recipe_recommender.config import DEFAULT_INTERACTIONS_PATH, DEFAULT_RECIPES_PATH
from recipe_recommender.core import RecipeRecommender
from recipe_recommender.evaluation import evaluate


def _parse_liked_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recipe recommender for recipes_visible_only.csv")
    parser.add_argument("--recipes-path", type=Path, default=DEFAULT_RECIPES_PATH, help="Path to recipes CSV.")
    parser.add_argument(
        "--interactions-path",
        type=Path,
        default=DEFAULT_INTERACTIONS_PATH,
        help="Path to interactions CSV (optional, may be empty).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results.")
    parser.add_argument("--model", choices=["hybrid", "tfidf", "bert"], default="hybrid")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERT and use TF-IDF only.")

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

    q = sub.add_parser("query", help="Recommend from free-text query.")
    q.add_argument("--text", required=True, type=str)

    s = sub.add_parser("similar", help="Recommend similar recipes to one recipe id.")
    s.add_argument("--recipe-id", required=True, type=int)

    l = sub.add_parser("liked", help="Recommend from list of liked recipe ids.")
    l.add_argument("--ids", required=True, type=str, help="Comma-separated recipe IDs.")

    m = sub.add_parser("message", help="Chatbot-style recommendation from a user message.")
    m.add_argument("--text", required=True, type=str)

    sub.add_parser("eval", help="Run evaluation suite.")
    return parser


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

    recommender = RecipeRecommender(use_bert=not args.no_bert)
    recommender.fit(args.recipes_path)
    print(f"Loaded {len(recommender.df)} visible recipes")
    print(f"BERT source: {recommender.bert_source}")

    if args.command == "query":
        out = recommender.recommend_for_query(args.text, top_k=args.top_k, model=args.model, **filters)
        print(out.to_string(index=False))
        return

    if args.command == "similar":
        out = recommender.recommend_similar(args.recipe_id, top_k=args.top_k, model=args.model, **filters)
        print(out.to_string(index=False))
        return

    if args.command == "liked":
        liked_ids = _parse_liked_ids(args.ids)
        out = recommender.recommend_for_liked(liked_ids, top_k=args.top_k, model=args.model, **filters)
        print(out.to_string(index=False))
        return

    if args.command == "message":
        out, parsed = recommender.recommend_from_message(args.text, top_k=args.top_k, model=args.model)
        print("Parsed preferences:", parsed)
        print(out.to_string(index=False))
        return

    if args.command == "eval":
        results = evaluate(
            recommender,
            interactions_path=args.interactions_path if args.interactions_path.is_file() else None,
            top_k=args.top_k,
        )
        print(results)
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
