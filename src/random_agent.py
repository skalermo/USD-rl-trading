from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


class RandomAgent:
    def __init__(self, env: DummyVecEnv, **kwargs):
        self.env = env

    def learn(self, **kwargs) -> None:
        pass

    def predict(self, _state: np.ndarray, **kwargs) -> np.ndarray:
        return self.env.action_space.sample(), None
