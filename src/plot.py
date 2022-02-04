from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt

from src.log_utils import process_logs
from plot_utils import aggregate_and_apply


plt.style.use('ggplot')
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

log_path = '../.data/logs'
all_logs = [f'{log_path}/{f}' for f in listdir(log_path) if isfile(join(log_path, f))]

models = ['A2C', 'PPO']
window_size = [10, 20, 30]
gamma = [0.7, 0.9, 0.99]
runs = [0, 1, 2]

for i, w in enumerate(window_size):
    for j, g in enumerate(gamma):
        for m in models:
            prefix = f'{m}_{w}_{g}'
            logs = [f for f in all_logs if f'{log_path}/{prefix}' in f]
            processed = []
            for log in logs:
                with open(log, 'r') as f:
                    processed.append(process_logs(f.read()))

            return_avg = aggregate_and_apply(processed, 'ep_rew_mean', lambda x: np.mean(x))
            return_std = aggregate_and_apply(processed, 'ep_rew_mean', lambda x: np.std(x))
            timesteps = aggregate_and_apply(processed, 'total_timesteps', lambda x: x[0])
            a = ax[i, j]
            # if env_id == 'MountainCarContinuous-v0':
            #     a.set_yscale('symlog')
            # else:
            #     a.set_yscale('linear')
            a.set_xlabel('Timestep')
            a.set_ylabel('Return')
            a.set_title(f'gamma={g} window={w}')

            plot = a.plot(timesteps, return_avg, label=m)
            c = plot[-1].get_color()
            a.fill_between(timesteps, np.asarray(return_avg) - np.asarray(return_std), np.asarray(return_avg) + np.asarray(return_std),
                            alpha=0.2, color=c)
            # a.set_title(env_id)

handles, labels = ax[-1][-1].get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles[::-1], labels[::-1], loc='center left')

# ax.legend()
plt.show()
fig.savefig(str(__file__).split('.')[0] + '0.pdf', bbox_inches='tight')
