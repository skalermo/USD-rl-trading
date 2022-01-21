import os
import sys
from typing import Callable, Type, Union

import gym_anytrading
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quantstats as qs

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.random_agent import RandomAgent
from src.log_utils import captured_output

# from src.stocks_env_custom import StocksEnvCustom


def _create_dirs(*paths: str):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
            

MODEL_TYPE = Type[Union[A2C, PPO, RandomAgent]]


def _str_to_class(classname: str) -> MODEL_TYPE:
    return getattr(sys.modules[__name__], classname)


def train_test_split(df: pd.DataFrame, split_point: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    assert 0.0 <= split_point <= 1.0
    df_train = df.iloc[:int(len(df) * split_point)]
    df_test = df.iloc[int(len(df) * split_point):]
    return df_train, df_test


def _test_loop(model: MODEL_TYPE, env: gym.Env) -> float:
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def run_quantstats(env: gym.Env, test_df: pd.DataFrame, window_size: int, output_path: str) -> None:
    net_worth = pd.Series(env.history['total_profit'], index=test_df.index[window_size + 1:])
    returns = net_worth.pct_change().iloc[1:]
    qs.reports.full(returns)
    qs.reports.html(returns, output=output_path)


def main():
    # total_timesteps = 1_000_000
    total_timesteps = 100

    df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
    train, test = train_test_split(df)

    window_size = 10

    def env_maker(df: pd.DataFrame, window_size: int) -> Callable[[], gym.Env]:
        start_index = window_size
        end_index = len(train)
        return lambda: gym.make( 'stocks-v0',
            df=df,
            window_size=window_size,
            frame_bound=(start_index, end_index)
        )

    models = {
        'RandomAgent': lambda verbose: RandomAgent(env=DummyVecEnv([env_maker(train, window_size)]), verbose=verbose),
        'A2C': lambda verbose: A2C(policy='MlpPolicy', env=DummyVecEnv([env_maker(train, window_size)]), verbose=verbose),
        'PPO': lambda verbose: PPO(policy='MlpPolicy', env=DummyVecEnv([env_maker(train, window_size)]), verbose=verbose),
    }

    # env_maker = lambda: StocksEnvCustom(df=df, window_size=window_size, frame_bound=(start_index, end_index))

    runs = 3
    # variable window size?
    # variable discount_factor?

    data_dir = '.data'
    logs_dir = f'{data_dir}/logs'
    models_dir = f'{data_dir}/models'
    quantstats_dir = f'{data_dir}/quantstats'
    _create_dirs(logs_dir, models_dir, quantstats_dir)
    
    for model_name, model_fn in models.items():
        for run in range(runs):
            model_path = f'{models_dir}/{model_name}_{run}.zip'
            print(f'Training {model_name} run {run}')
            if os.path.exists(model_path):
                print(f'Model {model_path} already exists, skipping')
                continue

            with captured_output() as (out, _):
                model = model_fn(verbose=1)
                model.learn(total_timesteps=total_timesteps)

            log_path = f'{logs_dir}/{model_name}_{run}.log'
            with open(log_path, 'w') as f:
                f.write(out.getvalue())
            model.save(model_path)

    qs.extend_pandas()

    for model_name, _ in models.items():
        # if model_name == 'RandomAgent':
        #     continue
        env = env_maker(test, window_size)()

        returns = []
        for run in range(runs):
            model_path = f'{models_dir}/{model_name}_0.zip'
            model = _str_to_class(model_name).load(model_path)
            return_ = _test_loop(model, env)
            returns.append(return_)
            quantstats_output_path = f'{quantstats_dir}/{model_name}_{run}.html'
            run_quantstats(env, test, window_size, quantstats_output_path)
        print(f'{model_name} returns (on avg): {sum(returns) / runs}')


    # plt.figure(figsize=(16, 6))
    # env.render_all()
    # plt.show()


if __name__ == '__main__':
    main()
