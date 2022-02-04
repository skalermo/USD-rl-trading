import os
import sys
from typing import Callable, Type, Union

import gym_anytrading
import gym
import pandas as pd
import numpy as np
import quantstats as qs
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO

from src.random_agent import RandomAgent
from src.log_utils import captured_output


def _create_dirs(*paths: str):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


MODEL_TYPE = Type[Union[A2C, PPO, RandomAgent, RecurrentPPO]]


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
        # env.render()
    return total_reward


def run_quantstats(env: gym.Env, test_df: pd.DataFrame, window_size: int, output_path: str) -> None:
    net_worth = pd.Series(env.history['total_profit'], index=test_df.index[window_size + 1:])
    returns = net_worth.pct_change().iloc[1:]
    qs.reports.full(returns)
    qs.reports.html(returns, output=output_path)


def main():
    df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
    train, test = train_test_split(df)

    def env_maker(df: pd.DataFrame, window_size: int) -> Callable[[], gym.Env]:
        start_index = window_size
        end_index = len(train)
        return lambda: gym.make('stocks-v0',
            df=df,
            window_size=window_size,
            frame_bound=(start_index, end_index)
        )

    models = {
        'RandomAgent': lambda window_size, *args: RandomAgent(env=env_maker(train, window_size)()),
        'A2C': lambda verbose, discount_factor, window_size: A2C(policy='MlpPolicy', env=env_maker(train, window_size)(), verbose=verbose, gamma=discount_factor),
        'PPO': lambda verbose, discount_factor, window_size: PPO(policy='MlpPolicy', env=env_maker(train, window_size)(), verbose=verbose, gamma=discount_factor),
        'RecurrentPPO': lambda verbose, discount_factor, window_size: RecurrentPPO(policy='MlpLstmPolicy', env=env_maker(train, window_size)(), verbose=verbose, gamma=discount_factor),
    }

    total_timesteps = 1_000_000
    runs = 3
    discount_factors = [0.99]
    window_sizes = [30]

    data_dir = '.data'
    logs_dir = f'{data_dir}/logs'
    models_dir = f'{data_dir}/models'
    quantstats_dir = f'{data_dir}/quantstats'
    _create_dirs(logs_dir, models_dir, quantstats_dir)

    for discount_factor in discount_factors:
        for window_size in window_sizes:
            for model_name, model_fn in models.items():
                for run in range(runs):
                    file_name = f'{model_name}_{window_size}_{discount_factor}_{run}'
                    model_path = f'{models_dir}/{file_name}.zip'
                    log_path = f'{logs_dir}/{file_name}.log'

                    print(f'Training {model_name} with window_size={window_size} and discount_factor={discount_factor} on run {run}')
                    if os.path.exists(model_path):
                        print(f'Model {model_path} already exists, skipping')
                        continue
                    if model_name == 'RandomAgent':
                        model = model_fn(window_size=window_size)
                        model.save(model_path)
                        continue

                    with captured_output() as (out, _):
                        model = model_fn(verbose=1, discount_factor=discount_factor, window_size=window_size)
                        model.learn(total_timesteps=total_timesteps)

                    with open(log_path, 'w') as f:
                        f.write(out.getvalue())
                    model.save(model_path)

    qs.extend_pandas()

    for discount_factor in discount_factors:
        for window_size in window_sizes:
            for model_name, _ in models.items():
                env = env_maker(test, window_size)()
                returns = []
                for run in range(runs):
                    file_name = f'{model_name}_{window_size}_{discount_factor}_{run}'
                    model_path = f'{models_dir}/{file_name}.zip'
                    model = _str_to_class(model_name).load(model_path)
                    for _ in range(10):
                        return_ = _test_loop(model, env)
                        returns.append(return_)
                    quantstats_output_path = f'{quantstats_dir}/{file_name}.html'
                    run_quantstats(env, test, window_size, quantstats_output_path)
                print(f'{model_name} gamme={discount_factor} window={window_size} returns (on avg): {np.mean(returns)} +- {np.std(returns)}')


if __name__ == '__main__':
    main()
