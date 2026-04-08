from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .core import RecipeRecommender


class EpsilonGreedyBandit:
    """
    Simple epsilon-greedy bandit for exploration vs exploitation.
    Use to choose among recommendation strategies (arms) from observed rewards
    (e.g., click, save, thumbs-up as 0/1 or graded reward).
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be at least 1.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()

        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        # Prefer arms with positive sample count, then break ties randomized among argmax.
        max_v = float(np.max(self.values))
        candidates = np.flatnonzero(self.values >= max_v - 1e-12)
        return int(self.rng.choice(candidates))

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("arm out of range")
        self.counts[arm] += 1
        n = float(self.counts[arm])
        self.values[arm] += (float(reward) - self.values[arm]) / n


def simulate_bandit(
    recommender: "RecipeRecommender",
    n_rounds: int = 200,
    epsilon: float = 0.1,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Simulate epsilon-greedy bandit choosing among 3 recommendation arms:
      Arm 0: random    — current company strategy (baseline)
      Arm 1: popular   — non-personalised popularity prior
      Arm 2: hybrid    — content-based hybrid (text + numeric + category)

    Reward proxy per round
    ----------------------
    A random "target" recipe is drawn (simulating a user's true preference).
    The selected arm produces top-10 recommendations.
    Reward = max cosine similarity between recommended items and the target
    in the shared hybrid embedding space. This rewards arms that return
    items semantically close to what the user actually wants.

    Returns a dict with per-round history and final bandit state.
    """
    rng = np.random.default_rng(random_seed)
    bandit = EpsilonGreedyBandit(n_arms=3, epsilon=epsilon, rng=rng)
    arm_names = ["random", "popular", "hybrid"]

    ids = recommender.df["id"].tolist()
    cumulative = 0.0
    history: dict[str, list] = {
        "round": [],
        "arm": [],
        "arm_name": [],
        "reward": [],
        "cumulative_reward": [],
    }

    for round_i in range(n_rounds):
        target_id = int(rng.choice(ids))
        target_idx = recommender.id_to_idx[target_id]
        arm = bandit.select_arm()

        if arm == 0:
            recs = recommender.recommend_random(
                top_k=10,
                random_state=int(rng.integers(0, 999_999)),
                exclude_recipe_ids=[target_id],
            )
        elif arm == 1:
            recs = recommender.recommend_popular(top_k=10, exclude_recipe_ids=[target_id])
        else:
            recs = recommender.recommend_similar(target_id, top_k=10, model="hybrid")

        rec_idxs = [
            recommender.id_to_idx[r]
            for r in recs["id"].tolist()
            if r in recommender.id_to_idx
        ]
        if rec_idxs:
            sims = (recommender.X_hybrid[rec_idxs] @ recommender.X_hybrid[target_idx]).ravel()
            reward = float(np.max(sims))
        else:
            reward = 0.0

        bandit.update(arm, reward)
        cumulative += reward

        history["round"].append(round_i + 1)
        history["arm"].append(arm)
        history["arm_name"].append(arm_names[arm])
        history["reward"].append(round(reward, 4))
        history["cumulative_reward"].append(round(cumulative, 4))

    counts = {arm_names[i]: int(bandit.counts[i]) for i in range(3)}
    values = {arm_names[i]: round(float(bandit.values[i]), 4) for i in range(3)}
    best_arm = arm_names[int(np.argmax(bandit.values))]

    # Compute arm selection rates for summary
    total = sum(counts.values())
    rates = {k: round(v / total, 3) for k, v in counts.items()}

    return {
        "n_rounds": n_rounds,
        "epsilon": epsilon,
        "best_arm": best_arm,
        "final_values": values,
        "final_counts": counts,
        "selection_rates": rates,
        "total_reward": round(cumulative, 4),
        "avg_reward_per_round": round(cumulative / n_rounds, 4),
        "history": history,
    }
