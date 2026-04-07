# Group Assignment Documentation

## 1) Project Goal (Business + Technical)

The company currently shows recipes randomly.  
Our goal is to replace that with a recommender system that improves relevance, engagement, and user satisfaction.

For the CEO (non-technical audience), the core message is:
- We built a working recommendation prototype end-to-end.
- We compared multiple recommendation families (not only one model).
- We can estimate business uplift vs random ranking.


## 2) Data We Have and What It Means

### Primary dataset for this project
- `recipes_visible_only.csv` (473 rows, 21 columns)
- This is the production-ready recipe catalog after filtering `visibility == True`.
- Key fields: `id`, `name`, `description`, `food_type`, `ingredient_groups`, nutrition fields, dietary flags, `cooking_time`.

### Supporting files and pipeline
- `recipes.py`: extracts recipe table from DB and exports raw/flattened CSVs.
- `recipes_clean_visible.py`: filters only visible recipes.
- `A6_Vashakidze_George.ipynb`: full recommender prototype with:
  - non-personalized baselines
  - content-based (BoW, TF-IDF, BERT)
  - hybrid weighting and grid search
  - chatbot query-to-recommendation flow
  - evaluation framework

### Important current limitation
- `interactions_train.csv` is empty.
- This means collaborative/personalized methods cannot be reliably trained on your own platform data yet.
- Therefore, for your dataset we should prioritize a strong cold-start recommendation logic first.


## 3) Assignment Requirement Mapping (from Group work.pdf)

The assignment expects breadth and integration. Our implementation plan covers:

1. Data exploration and preprocessing
2. Non-personalized recommenders (random, popular, optional demographic/category)
3. Content-based recommenders (BoW/TF-IDF/BERT, text normalization)
4. Hybrid recommender (weighted blend + weight tuning)
5. Chatbot-driven recommendation interface
6. Evaluation using ranking metrics (not accuracy only)
7. Clear interpretation + business added-value estimation
8. Deliverable: functioning prototype + CEO-friendly presentation


## 4) Recommendation Logic for `recipes_visible_only.csv`

This is the exact practical logic we should use now.

### Stage A: Candidate filtering (hard constraints)
Before scoring, filter by explicit user constraints:
- `is_vegan`, `is_vegetarian`, `is_gluten_free`, `is_lactose_free`
- `food_type` preference (Main Dish, Dessert, etc.)
- `max cooking time` from parsed `cooking_time`
- optional calorie/protein constraints

This guarantees recommendations satisfy explicit needs.

### Stage B: Candidate scoring (hybrid relevance)
Compute a weighted score:

`final_score = w_text * text_similarity + w_nutrition * nutrition_similarity + w_type * category_match + w_pop * popularity_prior`

Where:
- `text_similarity`: TF-IDF or BERT cosine similarity on `name + description + ingredients`
- `nutrition_similarity`: closeness to preferred macro profile (calories/protein/etc.)
- `category_match`: same or related `food_type`
- `popularity_prior`: optional small boost if engagement logs are available

Use the notebook's weight-search strategy to choose weights instead of hardcoding.

### Stage C: Re-ranking (quality of list)
After top-N scoring:
- diversify near-duplicates (same title pattern / nearly identical ingredients)
- keep variety across cuisine/type when possible
- guarantee at least one "safe" high-confidence item in top 3


## 5) Model Stack We Should Report

For the final group submission, present these model families:

1. **Random baseline** (current company strategy)
2. **Popular baseline** (if interactions exist)
3. **Content TF-IDF**
4. **Content BERT (MiniLM)**
5. **Hybrid (best weights from search)**
6. **Collaborative filtering (SVD)** once real interaction logs are available

Note: Until interactions are non-empty, CF should be clearly marked as "future/limited."


## 6) Evaluation Plan

### Offline metrics (ranking)
- `HitRate@K`
- `Recall@K`
- `MRR@K`
- `NDCG@K` (add for final report)
- Coverage and diversity (already in notebook)

### Split strategy
- Use time-based split for interactions (when available).
- Keep same evaluation protocol across all models for fairness.

### If interactions remain unavailable
Use a two-part evaluation:
1. **Constraint satisfaction rate** (did results obey vegan/time/etc. constraints?)
2. **Human relevance scoring** (top-10 judged by team members with a rubric)

This is acceptable for cold-start proof of concept and still defensible to the CEO.


## 7) Chatbot Recommendation Mode

We already have manual and automated chatbot logic in notebook.

Production behavior should be:
1. Parse free text into structured intent:
   - meal type
   - dietary restrictions
   - ingredients liked/disliked
   - time budget
2. Apply hard filters
3. Rank with hybrid score
4. Return top recipes with short explanation ("recommended because...")

This is critical for user trust and presentation quality.


## 8) Business Added-Value Estimation (for CEO slide)

Estimate uplift vs random with scenario assumptions:

- Base traffic: monthly recommendation impressions
- Current random CTR (or proxy)
- Expected CTR uplift from best model
- Conversion from click to save/order/cook action

Report:
- Conservative, base, optimistic scenarios
- Incremental clicks/actions per month
- Expected impact range (not single-point claim)


## 9) Implementation Plan (Team Execution)

### Workstream A - Data + Features
- Finalize parsing of `ingredient_groups` to ingredient text
- Parse `cooking_time` to minutes
- Build canonical item text field

### Workstream B - Recommenders
- Implement TF-IDF + BERT retrieval for `recipes_visible_only.csv`
- Implement hybrid scoring + grid search
- Add rule-based constraints and reranking

### Workstream C - Evaluation + Reporting
- Run unified metric table
- Add constraint-satisfaction and diversity
- Generate concise interpretation and business impact estimate

### Workstream D - Demo + Presentation
- Notebook walkthrough
- Chatbot query demo
- CEO-facing deck with 1-page executive summary


## 10) Definition of Done

At project completion, we must be able to:

1. Input user preferences (or free-text query)
2. Recommend top-N recipes from `recipes_visible_only.csv`
3. Explain why those recipes were chosen
4. Show model comparison vs random baseline
5. Present estimated business value in clear non-technical terms


## 11) Immediate Next Steps

1. Use `recipes_visible_only.csv` as the default catalog in the notebook.
2. Add a parsing utility for `ingredient_groups` into a clean ingredient text column.
3. Run TF-IDF + BERT + hybrid on this dataset specifically.
4. Add final comparison table and save outputs for presentation.
5. Keep CF section present but labeled as pending richer interaction data.


## 12) Risks and Mitigations

- **Risk:** No interaction data for personalization  
  **Mitigation:** Focus on content + constraints + chatbot; start event logging now.

- **Risk:** Small catalog (473 recipes) can overfit rankings  
  **Mitigation:** diversity reranking, qualitative review, periodic retraining.

- **Risk:** Large artifacts in git cause push failures  
  **Mitigation:** keep Food.com raw files ignored (`.gitignore`), store only code/docs.


---

This document defines the assignment approach using your current assets and ensures the final system can recommend recipes directly from `recipes_visible_only.csv`.
