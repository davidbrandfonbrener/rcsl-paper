import fire
import itertools
import os
from subprocess import Popen

def get_overrides(sweep_idx):
    grid = {
        "env_name": ['maze2d-umaze-v1'],
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_idx - 1]  # slurm var will start from 1

    overrides = {
        "sweep_idx": sweep_idx,
        "env_name": step_grid["env_name"],
    }

    return overrides


def run_experiment(overrides):
    dir = '/scratch/db3854/jax_continuous_rl/experiments'
    os.chdir(dir)

    cmd = ['python', 'train_rvs.py']
    for k, v in overrides.items():
        cmd.append(f' --{k} {v}')

    env = os.environ.copy()
    p = Popen(cmd, env=env)
    p.communicate()

def main(sweep_idx):
    overrides = get_overrides(sweep_idx)
    run_experiment(overrides)

if __name__ == "__main__":
    fire.Fire(main)