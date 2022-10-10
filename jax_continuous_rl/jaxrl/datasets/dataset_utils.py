from calendar import day_abbr
from typing import Tuple

import gym

import d4rl
import numpy as np

from jaxrl.datasets.awac_dataset import AWACDataset
from jaxrl.datasets.d4rl_dataset import D4RLDataset, DatasetFromDict
from jaxrl.datasets.rvs_d4rl_dataset import D4rlRvsDataset
from jaxrl.datasets.dataset import Dataset, normalize
from jaxrl.datasets.rl_unplugged_dataset import RLUnpluggedDataset
from jaxrl.utils import make_env
from jaxrl.wrappers.normalize_reward  import NormalizeReward


def make_env_and_dataset(rvs_data: bool, env_name: str, seed: int,
                         dataset_name: str,
                         use_avg_returns, use_max_length,
                         discount, transform_reward,
                         video_save_folder: str) -> Tuple[gym.Env, Dataset]:
    env = make_env(env_name, seed, video_save_folder)

    if rvs_data:
        if dataset_name == 'd4rl':
            dataset_dict = d4rl.qlearning_dataset(env)
            if 'antmaze' in env_name:
                dataset_dict['rewards'] -= 1.0
        elif dataset_name == 'mine':
            dataset_dict = dict(np.load(f'../data/{env_name}.npz'))
        else:
            raise NotImplementedError

        dataset = DatasetFromDict(dataset_dict)
        shift, scale = normalize(dataset)
        scale = scale / 1000.
        dataset_dict['rewards'] = (dataset_dict['rewards'] -  shift) * scale
        
        env = NormalizeReward(env, shift, scale)
        dataset = D4rlRvsDataset(dataset_dict, 
                            use_avg_returns,
                            use_max_length,
                            discount,
                            transform_reward)

    else: 
        if 'd4rl' in dataset_name:
            dataset = D4RLDataset(env)
            if 'antmaze' in env_name:
                dataset.rewards -= 1.0
        
        elif 'awac' in dataset_name:
            dataset = AWACDataset(env_name)
        elif 'rl_unplugged' in dataset_name:
            dataset = RLUnpluggedDataset(env_name.replace('-', '_'))
        elif 'mine' in dataset_name:
            dataset = dict(np.load(f'../data/{env_name}.npz'))
            dataset = DatasetFromDict(dataset)
        else:
            raise NotImplementedError(f'{dataset_name} is not available!')

        shift, scale = normalize(dataset)

    return env, dataset
