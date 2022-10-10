import fire
from utils import run_experiment, write_slurm_file

dir = '/scratch/db3854/jax_continuous_rl/experiments'
script_name = 'train_offline.py'
slurm_name = 'ddpg_sweep'

grid = {
    "name": ['ddpg'],
    "config": ['configs/ddpg_default.py'],
    "env_name": ['pen-cloned-v1', 'pen-expert-v1', 'hammer-human-v1', 'hammer-cloned-v1', 'hammer-expert-v1'],
    #['halfcheetah-medium-replay-v2', 'antmaze-umaze-v2', 'antmaze-medium-play-v2', 'pen-human-v1'],
    
    "config.hidden_dims": [(256,256)],
    "config.actor_lr": [0.0003],
    "config.bc_alpha": [1.0, 2.5, 5.0],
    
    "seed": [0,1,2],
    "save_model": ['False'],
}

# grid = {
#     "name": ['ddpg'],
#     "config": ['configs/ddpg_default.py', ],
#     "env_name": [ 'point_mass-stitch-easy', 'point_mass-stitch-stitch',],
#         #'point_mass-stitch-easy', 'point_mass-stitch-stitch'],
#     #                 'point_mass-open-offset', 'point_mass-dense-offset',
#     #                 'point_mass-wideinit-normal','point_mass-widedense-normal'],
#     "dataset_name": ['mine'],
    
#     "config.hidden_dims": [(256,256)],
#     "config.actor_lr": [0.0003],
#     "config.critic_lr": [0.0003],
#     "config.bc_alpha": [1.0, 2.5, 5.0],
    
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