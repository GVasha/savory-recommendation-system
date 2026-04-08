from __future__ import annotations

import numpy as np


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
