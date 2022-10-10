import fire
from utils import run_experiment, write_slurm_file

dir = '/scratch/db3854/jax_continuous_rl/experiments'
script_name = 'train_offline.py'
slurm_name = 'rvs_sweep'

grid = {
    "name": ['rvs'],
    "config": ['configs/rvs_default.py'],
    "env_name": ['pen-cloned-v1', 'pen-expert-v1', 'hammer-human-v1', 'hammer-cloned-v1', 'hammer-expert-v1'],
    #['antmaze-umaze-v2', 'antmaze-medium-play-v2', 'halfcheetah-medium-replay-v2', 'pen-human-v1'],
    "avg_returns": ['True', 'False'],

    "discount": [1.0],
    "eval_outcome": [0.8, 1.0, 1.2],

    "config.hidden_dims": [ (256,256)], # (1024,1024),
    "config.actor_lr": [0.001],

    "seed": [0,1,2],
    "save_model": ['False'],
}


# grid = {
#    "name": ['rvs'],
#     "config": ['configs/rvs_default.py'],

#     "discount": [1.0],
#     "eval_outcome": [0.8, 1.0, 1.2],
#     "env_name": [ 'point_mass-stitch-easy', 'point_mass-stitch-stitch',],
#         #'point_mass-open-offset', 'point_mass-dense-offset'],
#         #'point_mass-ring_of_fire-offset'],
#                     # 'point_mass-wideinit-normal', 'point_mass-widedense-normal'],
#     "dataset_name": ['mine'],

#     "avg_returns": ['True', 'False'],
    
#     "config.hidden_dims": [(256,256)],
#     "config.actor_lr": [0.001],
    
#     "seed": [0,1,2],
#     "save_model": ['True'],
# }


def main(arg):
    if arg == 0:
        write_slurm_file(grid, slurm_name)
    else:
        run_experiment(arg, grid, dir, script_name)

if __name__ == "__main__":
    fire.Fire(main)