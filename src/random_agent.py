import pickle

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


class RandomAgent:
    def __init__(self, env: DummyVecEnv, **kwargs):
        self.env = env

    def learn(self, **kwargs) -> None:
        pass

    def predict(self, _state: np.ndarray, **kwargs) -> np.ndarray:
        return self.env.action_space.sample(), None

    def save(self, path):
        self.create_nn = None
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError:
            return None
        return obj
