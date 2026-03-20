import numpy as np

class RandomAgent:
    """Uniformly random swap selection. Lower bound baseline."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None