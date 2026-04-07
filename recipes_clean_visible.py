"""
Filter recipe exports to only rows where visibility is true (public / visible).

Reads recipes_raw.csv by default; override paths with CLI args or constants below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_CSV = Path(__file__).resolve().parent / "recipes_raw.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "recipes_visible_only.csv"


def normalize_bool(val: Any) -> bool | None:
    """Map common DB/CSV representations to bool; None stays missing."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if not s:
        return None
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def filter_visible_only(df: pd.DataFrame, column: str = "visibility") -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(
            f"Column {column!r} not found. Available: {list(df.columns)}"
        )

    normalized = df[column].map(normalize_bool)
    # Keep rows that are explicitly True; drop False and unknown (None)
    mask = normalized == True
    return df.loc[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only recipes with visibility=True.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=INPUT_CSV,
        help="Input CSV path (default: recipes_raw.csv next to this script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path (default: recipes_visible_only.csv)",
    )
    parser.add_argument(
        "-c",
        "--column",
        default="visibility",
        help="Boolean visibility column name (default: visibility)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    before = len(df)
    filtered = filter_visible_only(df, column=args.column)
    after = len(filtered)

    filtered.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Read {before} rows from {args.input}")
    print(f"Kept {after} rows with {args.column}=True -> {args.output}")


if __name__ == "__main__":
    main()
