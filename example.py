from typing import Callable

import gym
from stable_baselines3 import A2C, PPO
import matplotlib.pyplot as plt
import pandas as pd
from gym_anytrading.datasets import STOCKS_GOOGL

from sb3_contrib import RecurrentPPO
from src.random_agent import RandomAgent


def train_test_split(df: pd.DataFrame, split_point: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    assert 0.0 <= split_point <= 1.0
    df_train = df.iloc[:int(len(df) * split_point)]
    df_test = df.iloc[int(len(df) * split_point):]
    return df_train, df_test


def main():
    df = STOCKS_GOOGL.copy()
    train, test = train_test_split(df)

    def env_maker(df: pd.DataFrame, window_size: int) -> Callable[[], gym.Env]:
        start_index = window_size
        end_index = len(train)
        return lambda: gym.make('stocks-v0',
            df=df,
            window_size=window_size,
            frame_bound=(start_index, end_index)
        )
    env = env_maker(test, window_size=30)()
    obs = env.reset()

    models = ['RandomAgent', 'A2C', 'PPO', 'RecurrentPPO']
    model_classes = [RandomAgent, A2C, PPO, RecurrentPPO]

    saved_model_path = f'./.data/models/{models[-1]}_30_0.99_0.zip'
    model = model_classes[-1].load(saved_model_path)
    if model is None:
        print('Model not found. Please train the model first.')
        model = RecurrentPPO(env, verbose=True)
        model.learn(total_timesteps=1000, log_interval=100)
        model.save(saved_model_path)

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

    plt.cla()
    env.render_all()
    plt.show()


if __name__ == '__main__':
    main()

