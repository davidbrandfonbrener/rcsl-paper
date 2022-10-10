import fire
from utils import run_experiment, write_slurm_file

dir = '/scratch/db3854/jax_continuous_rl/experiments'
script_name = 'train_offline.py'
slurm_name = 'iql_sweep'

grid = {
    "name": ['iql'],
    "config": ['configs/iql_default.py'],
    "env_name": ['pen-cloned-v1', 'pen-expert-v1', 'pen-human-v1', 'halfcheetah-medium-replay-v2', 'antmaze-umaze-v2', 'antmaze-medium-play-v2', 'antmaze-large-play-v2'],
    
    "config.hidden_dims": [(256,256)],
    "config.actor_lr": [0.0003],
    "config.expectile": [0.5, 0.7, 0.9],
    "config.temperature": [10.0],
    
    "seed": [0,1,2],
    "save_model": ['False'],
}

# grid = {
#     "name": ['iql'],
#     "config": ['configs/iql_default.py'],
#     "env_name": ['point_mass-stitch-easy', 'point_mass-stitch-stitch',
#                       'point_mass-ring_of_fire-offset',
#                       'point_mass-open-offset', 'point_mass-dense-offset',
#                       'point_mass-wideinit-normal','point_mass-widedense-normal'],
#     "dataset_name": ['mine'],
    
#     "config.hidden_dims": [(256,256)],
#     "config.actor_lr": [0.0003],
#     #"config.critic_lr": [0.0003, 0.0001],
#     "config.expectile": [0.5, 0.7, 0.9],
#     "config.temperature": [10.0],
    
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