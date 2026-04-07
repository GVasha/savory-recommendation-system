# Recipe Recommender Project (Visible Catalog)

This project implements an end-to-end recommender on top of:
- `recipes_visible_only.csv` (primary catalog)

It supports:
- content recommendations (TF-IDF / BERT)
- hybrid recommendations (text + numeric + category)
- chatbot-style message parsing
- cold-start quality evaluation
- interaction-based evaluation when interactions are available

## Files

- `recipe_recommender/core.py` - main recommender class
- `recipe_recommender/preprocessing.py` - parsing/cleaning helpers
- `recipe_recommender/chatbot.py` - message -> preference parser
- `recipe_recommender/evaluation.py` - evaluation utilities
- `run_recommender.py` - CLI entrypoint

## Quick Start

```bash
python run_recommender.py query --text "quick high-protein dinner with chicken" --top-k 10
```

## Example Commands

Query-based:

```bash
python run_recommender.py query --text "vegan soup under 30 minutes" --vegan --food-type Soup --max-minutes 30
```

Similar recipe:

```bash
python run_recommender.py similar --recipe-id 317 --top-k 10
```

From liked recipes:

```bash
python run_recommender.py liked --ids "317,314,353" --top-k 10
```

Chatbot-style:

```bash
python run_recommender.py message --text "I want a lactose-free high-protein main dish under 40 minutes with chicken"
```

Evaluation:

```bash
python run_recommender.py eval --top-k 10
```

If BERT causes environment issues, disable it:

```bash
python run_recommender.py query --text "quick dinner" --no-bert
```
