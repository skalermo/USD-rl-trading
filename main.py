import gym_anytrading
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quantstats as qs

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# from src.stocks_env_custom import StocksEnvCustom


def main():
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

    env = DummyVecEnv([env_maker])

    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

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
