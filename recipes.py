import json
import re
from typing import Any

import pandas as pd
import psycopg2


# =========================
# 1) DB CONNECTION SETTINGS
# =========================
DB_CONFIG = {
   
}

# Optional: if you already know the exact table, put it here, e.g. "recipes"
TABLE_NAME = None   # or "recipes"


# =========================
# 2) HELPERS
# =========================
def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def parse_time_to_minutes(value: Any) -> Any:
    if value is None:
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

    total = hours * 60 + minutes
    return total if total > 0 else None


def extract_ingredient_texts(value: Any) -> str:
    """
    Handles cases like:
    - list of strings
    - list of dicts with 'text'
    - json string
    """
    if value is None:
        return ""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except Exception:
            return value

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
        return ", ".join(parts)

    return safe_json_dumps(value)


def extract_instructions_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except Exception:
            return value

    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    return safe_json_dumps(value)


def normalize_bool(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"true", "t", "1", "yes"}:
        return True
    if s in {"false", "f", "0", "no"}:
        return False
    return val


# =========================
# 3) DATABASE INTROSPECTION
# =========================
def get_candidate_tables(conn) -> pd.DataFrame:
    query = """
    SELECT
        c.table_schema,
        c.table_name,
        string_agg(c.column_name, ', ' ORDER BY c.ordinal_position) AS columns
    FROM information_schema.columns c
    WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
    GROUP BY c.table_schema, c.table_name
    ORDER BY c.table_schema, c.table_name;
    """
    return pd.read_sql(query, conn)


def choose_recipe_table(conn) -> str:
    tables = get_candidate_tables(conn)

    best_score = -1
    best_full_name = None

    for _, row in tables.iterrows():
        schema = row["table_schema"]
        table = row["table_name"]
        columns = row["columns"].lower()

        score = 0

        # Table name hints
        if "recipe" in table.lower():
            score += 10
        if "recipes" == table.lower():
            score += 20

        # Column hints
        for token, pts in [
            ("id", 1),
            ("name", 3),
            ("title", 3),
            ("description", 3),
            ("ingredients", 5),
            ("ingredient_groups", 5),
            ("ingredientstext", 5),
            ("instructions", 5),
            ("author", 3),
            ("food_type", 3),
            ("youtube_url", 2),
            ("content_creator_id", 2),
            ("calories", 2),
            ("carbs", 2),
            ("proteins", 2),
            ("fats", 2),
            ("time", 2),
            ("visibility", 2),
        ]:
            if token in columns:
                score += pts

        if score > best_score:
            best_score = score
            best_full_name = f'"{schema}"."{table}"'

    if not best_full_name:
        raise RuntimeError("Could not find a likely recipes table.")

    return best_full_name


def get_table_columns(conn, full_table_name: str) -> pd.DataFrame:
    schema_name, table_name = full_table_name.replace('"', "").split(".")
    query = f"""
    SELECT
        column_name,
        data_type,
        udt_name
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """
    return pd.read_sql(query, conn, params=[schema_name, table_name])


# =========================
# 4) EXPORT DATA
# =========================
def fetch_all_rows(conn, full_table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM {full_table_name};"
    return pd.read_sql(query, conn)


def flatten_recipes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Convert complex columns to readable strings if needed
    for col in out.columns:
        sample_non_null = next((x for x in out[col] if pd.notna(x)), None)

        if isinstance(sample_non_null, (dict, list)):
            out[col] = out[col].apply(safe_json_dumps)

    # Build helpful flattened fields if these columns exist
    if "ingredientsText" in out.columns:
        out["ingredients_text_flat"] = out["ingredientsText"].apply(extract_ingredient_texts)
    elif "ingredientstext" in out.columns:
        out["ingredients_text_flat"] = out["ingredientstext"].apply(extract_ingredient_texts)

    if "ingredients" in out.columns:
        out["ingredients_flat"] = out["ingredients"].apply(extract_ingredient_texts)

    if "instructions" in out.columns:
        out["instructions_flat"] = out["instructions"].apply(extract_instructions_text)

    if "Time" in out.columns:
        out["time_minutes"] = out["Time"].apply(parse_time_to_minutes)
    elif "time" in out.columns:
        out["time_minutes"] = out["time"].apply(parse_time_to_minutes)

    # Normalize likely boolean columns
    for col in ["is_vegan", "is_vegetarian", "is_lactose_free", "is_gluten_free", "visibility"]:
        if col in out.columns:
            out[col] = out[col].apply(normalize_bool)

    return out


# =========================
# 5) MAIN
# =========================
def main():
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        if TABLE_NAME:
            full_table_name = TABLE_NAME
            if "." not in full_table_name:
                full_table_name = f'"public"."{TABLE_NAME}"'
        else:
            full_table_name = choose_recipe_table(conn)

        print(f"Using table: {full_table_name}")

        columns_df = get_table_columns(conn, full_table_name)
        print("\nColumns found:")
        print(columns_df)

        raw_df = fetch_all_rows(conn, full_table_name)
        print(f"\nRows fetched: {len(raw_df)}")

        raw_df.to_csv("recipes_raw.csv", index=False, encoding="utf-8-sig")
        print("Saved: recipes_raw.csv")

        flat_df = flatten_recipes(raw_df)
        flat_df.to_csv("recipes_flattened.csv", index=False, encoding="utf-8-sig")
        print("Saved: recipes_flattened.csv")

    finally:
        conn.close()


if __name__ == "__main__":
    main()