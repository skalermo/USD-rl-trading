import gym_anytrading
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quantstats as qs

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.random_agent import RandomAgent

# from src.stocks_env_custom import StocksEnvCustom


def train_test_split(df: pd.DataFrame, split_point: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    assert 0.0 <= split_point <= 1.0
    df_train = df.iloc[:int(len(df) * split_point)]
    df_test = df.iloc[int(len(df) * split_point):]
    return df_train, df_test


def main():
    # total_timesteps = 1_000_000
    total_timesteps = 1_000

    models = {
        'RandomAgent': lambda verbose: RandomAgent(env=DummyVecEnv([env_maker]), verbose=verbose),
        'A2C': lambda verbose: A2C(policy='MlpPolicy', env=DummyVecEnv([env_maker]), verbose=verbose),
        'PPO': lambda verbose: PPO(policy='MlpPolicy', env=DummyVecEnv([env_maker]), verbose=verbose),
    }

    df = gym_anytrading.datasets.STOCKS_GOOGL.copy()

    window_size = 10
    start_index = window_size
    end_index = len(df)

    env_maker = lambda: gym.make(
        'stocks-v0',
        df=df,
        window_size=window_size,
        frame_bound=(start_index, end_index)
    )

    # env_maker = lambda: StocksEnvCustom(df=df, window_size=window_size, frame_bound=(start_index, end_index))

    runs = 3

    for model_name, model_fn in models.items():
        for run in range(runs):
            print(f'{model_name} run {run}')
            model = model_fn(verbose=1)
            model.learn(total_timesteps=total_timesteps)

    env = DummyVecEnv([env_maker])
    env = env_maker()
    observation = env.reset()

    while True:
        observation = observation[np.newaxis, ...]

        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)

        # env.render()
        if done:
            print("info:", info)
            break

    plt.figure(figsize=(16, 6))
    env.render_all()
    plt.show()

    qs.extend_pandas()

    net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index + 1:end_index])
    returns = net_worth.pct_change().iloc[1:]

    qs.reports.full(returns)
    qs.reports.html(returns, output='a2c_quantstats.html')


if __name__ == '__main__':
    main()
