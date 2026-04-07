import ast
import re
from typing import Any

import numpy as np
import pandas as pd


def normalize_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def parse_time_to_minutes(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip().lower()
    if not s:
        return None

    hours = 0
    minutes = 0

    hour_match = re.search(r"(\d+)\s*hour", s)
    minute_match = re.search(r"(\d+)\s*minute", s)

    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))

    # Handle plain numbers like "45"
    if hours == 0 and minutes == 0 and s.isdigit():
        minutes = int(s)

    total = hours * 60 + minutes
    return float(total) if total > 0 else None


def safe_literal_eval(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return value


def ingredient_groups_to_text(value: Any) -> str:
    parsed = safe_literal_eval(value)

    if parsed is None:
        return ""

    if isinstance(parsed, dict):
        tokens: list[str] = []
        for _, group_items in parsed.items():
            if isinstance(group_items, list):
                for item in group_items:
                    if isinstance(item, dict) and item.get("text"):
                        tokens.append(str(item["text"]))
                    elif isinstance(item, str):
                        tokens.append(item)
        return " ".join(tokens)

    if isinstance(parsed, list):
        tokens = []
        for item in parsed:
            if isinstance(item, dict) and item.get("text"):
                tokens.append(str(item["text"]))
            elif isinstance(item, str):
                tokens.append(item)
        return " ".join(tokens)

    return str(parsed)


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    name = df.get("name", pd.Series([""] * len(df), index=df.index)).fillna("")
    desc = df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("")
    food_type = df.get("food_type", pd.Series([""] * len(df), index=df.index)).fillna("")
    ing = df.get("ingredient_text", pd.Series([""] * len(df), index=df.index)).fillna("")
    return (name + " " + desc + " " + food_type + " " + ing).map(normalize_text)
