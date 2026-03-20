from __future__ import annotations

from pathlib import Path

import numpy as np


class PPOAgent:
    """Thin inference wrapper around a Stable-Baselines3 PPO policy."""

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(str(model_path), device=device)

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
