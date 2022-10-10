import numpy as np
import collections
from tqdm import tqdm

RvsBatch = collections.namedtuple('Batch',
                ['observations', 'actions', 'outcomes'])
Timestep = collections.namedtuple('Timestep',
                ['observation', 'action', 'outcome', 'done'])

def split_into_trajectories(observations, actions, rewards, dones_float):
    trajs = [[]]
    for i in tqdm(range(len(observations))):
        trajs[-1].append(Timestep(observations[i], actions[i], 
                            rewards[i], dones_float[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])
    print('Num trajectories: ', len(trajs))
    return trajs

def merge_trajectories(trajs):
    observations = []
    actions = []
    outcomes = []
    dones_float = []

    for traj in trajs:
        for timestep in traj:
            observations.append(timestep.observation)
            actions.append(timestep.action)
            outcomes.append(timestep.outcome)
            dones_float.append(timestep.done)

    return np.stack(observations), np.stack(actions), np.stack(
            outcomes), np.stack(dones_float)


class RvsDataset(object):
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 outcomes: np.ndarray, dones_float: np.ndarray):
        self.observations = observations
        self.actions = actions
        self.outcomes = outcomes
        self.dones_float = dones_float
        self.size = len(observations)
        assert self.size == len(actions) and self.size == len(outcomes)

    def sample(self, batch_size: int):
        indx = np.random.randint(self.size, size=batch_size)
        return RvsBatch(observations=self.observations[indx],
                        actions=self.actions[indx],
                        outcomes=self.outcomes[indx])

    def train_validation_split(self, train_fraction: float):
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.outcomes, self.dones_float,)

        if len(trajs) == 1:
            train_size = int(train_fraction * self.size)
            train_dataset = RvsDataset(self.observations[:train_size], 
                                        self.actions[:train_size],
                                        self.outcomes[:train_size], 
                                        self.dones_float[:train_size])
            valid_dataset = RvsDataset(self.observations[train_size:], 
                                        self.actions[train_size:],
                                        self.outcomes[train_size:], 
                                        self.dones_float[train_size:])                        
        else:
            train_size = int(train_fraction * len(trajs))
            np.random.shuffle(trajs)

            (train_observations, train_actions, train_outcomes,
                train_dones_float) = merge_trajectories(trajs[:train_size])

            (valid_observations, valid_actions, valid_outcomes,
                valid_dones_float) = merge_trajectories(trajs[train_size:])

            train_dataset = RvsDataset(train_observations, train_actions,
                                        train_outcomes, train_dones_float)
            valid_dataset = RvsDataset(valid_observations, valid_actions,
                                        valid_outcomes, valid_dones_float)

        print('Split sizes: ', train_dataset.size, valid_dataset.size)
        return train_dataset, valid_dataset


def compute_traj_returns(traj, discount):
    traj_deque = collections.deque(traj)
    returns = [traj_deque.pop().outcome]
    while len(traj_deque) > 0:
        step = traj_deque.pop()
        returns.append(step.outcome + discount * returns[-1])
    returns = returns[::-1]
    return returns

def compute_returns(observations, actions, rewards, dones_float, discount):
    trajs = split_into_trajectories(observations, actions, rewards, dones_float)
    returns = []
    for traj in tqdm(trajs):
        traj_returns = compute_traj_returns(traj, discount)
        returns.extend(traj_returns)
    return np.expand_dims(np.array(returns, dtype=np.float32), 1)


def compute_traj_avg_returns(returns, max_length = None):
    if max_length is None:
        max_length = len(returns)
    avg_returns = []
    for i, r in enumerate(returns):
        avg_returns.append(r / (max_length - i))
    return avg_returns

def compute_avg_returns(observations, actions, rewards, dones_float,
                            use_max_length=True):
    trajs = split_into_trajectories(observations, actions, rewards, dones_float)
    if use_max_length:
        max_length = np.max([len(traj) for traj in trajs])
    else:
        max_length = None
    
    avg_returns = []
    for traj in tqdm(trajs):
        traj_returns = compute_traj_returns(traj, 1.0)
        avg_traj_returns = compute_traj_avg_returns(traj_returns, max_length)
        avg_returns.extend(avg_traj_returns)
    return np.expand_dims(np.array(avg_returns, dtype=np.float32), 1)


class D4rlRvsDataset(RvsDataset):

    def __init__(self,
                 dataset,
                 use_avg_returns: bool = False,
                 use_max_length: bool = False,
                 discount: float = 1.0,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transform_reward: bool = False):

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or \
                dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        # Compute returns as outcomes
        observations = dataset['observations'].astype(np.float32)
        actions = dataset['actions'].astype(np.float32)
        rewards = dataset['rewards'].astype(np.float32)
        dones_float = dones_float.astype(np.float32)

        if transform_reward:
            rewards = rewards / 1000.

        if use_avg_returns:
            returns = compute_avg_returns(observations, actions,
                                    rewards, dones_float, use_max_length)
        else:
            returns = compute_returns(observations, actions,
                                    rewards, dones_float, discount)

        super().__init__(observations, actions, returns, dones_float)
