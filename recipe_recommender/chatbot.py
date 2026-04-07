import re
from dataclasses import dataclass, field


@dataclass
class QueryPreferences:
    query_text: str
    max_minutes: int | None = None
    food_type: str | None = None
    is_vegan: bool | None = None
    is_vegetarian: bool | None = None
    is_gluten_free: bool | None = None
    is_lactose_free: bool | None = None
    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)


def parse_user_message(message: str) -> QueryPreferences:
    text = message.strip()
    lower = text.lower()
    prefs = QueryPreferences(query_text=text)

    # Time constraints like "under 30 minutes", "max 45 min"
    minute_match = re.search(
        r"(?:under|max(?:imum)?|within)\s*(\d{1,3})\s*(?:minute|min|minutes)", lower
    )
    if minute_match:
        prefs.max_minutes = int(minute_match.group(1))

    # Dietary constraints
    if "vegan" in lower:
        prefs.is_vegan = True
        prefs.is_vegetarian = True
    elif "vegetarian" in lower:
        prefs.is_vegetarian = True

    if "gluten free" in lower or "gluten-free" in lower:
        prefs.is_gluten_free = True
    if "lactose free" in lower or "lactose-free" in lower or "no dairy" in lower:
        prefs.is_lactose_free = True

    # Food type hints
    food_type_map = {
        "dessert": "Dessert",
        "main dish": "Main Dish",
        "main": "Main Dish",
        "soup": "Soup",
        "appetizer": "Appetizer",
        "breakfast": "Breakfast",
        "snack": "Snack",
        "sauce": "Sauce",
        "side dish": "Side Dish",
    }
    for k, v in food_type_map.items():
        if k in lower:
            prefs.food_type = v
            break

    # Simple include/exclude extraction
    for phrase in re.findall(r"(?:with|include|containing)\s+([a-zA-Z\s,]+)", lower):
        prefs.include_terms.extend([x.strip() for x in phrase.split(",") if x.strip()])

    for phrase in re.findall(r"(?:without|exclude|no)\s+([a-zA-Z\s,]+)", lower):
        prefs.exclude_terms.extend([x.strip() for x in phrase.split(",") if x.strip()])

    return prefs
