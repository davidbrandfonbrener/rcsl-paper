import fire
from utils import run_experiment, write_slurm_file

dir = '/scratch/db3854/jax_continuous_rl/experiments'
script_name = 'train_offline.py'
slurm_name = 'joint_rvs_sweep'

grid = {
   "name": ['joint_rvs'],
    "config": ['configs/joint_rvs_default.py'],

    "discount": [1.0],
    "eval_outcome": [0.5, 0.8, 0.99],
    "env_name": [ 'point_mass-stitch-easy', 'point_mass-stitch-stitch',
                    'point_mass-open-offset', 'point_mass-dense-offset',
                    'point_mass-wideinit-normal',],
    "dataset_name": ['mine'],

    "avg_returns": ['True', 'False'],
    
    "config.hidden_dims": [(256,256)],
    "config.actor_lr": [0.001],
    
    "seed": [0,1,2],
    "save_model": ['True'],
}

def main(arg):
    if arg == 0:
        write_slurm_file(grid, slurm_name)
    else:
        run_experiment(arg, grid, dir, script_name)

if __name__ == "__main__":
    fire.Fire(main)