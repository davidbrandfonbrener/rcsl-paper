import itertools
import os
from subprocess import Popen

def run_experiment(sweep_idx, grid, dir, script_name):
    sweep_list = list(dict(zip(grid.keys(), values)) 
                    for values in itertools.product(*grid.values()))
    overrides = sweep_list[sweep_idx - 1]

    os.chdir(dir)

    cmd = ['python', script_name]
    for k, v in overrides.items():
        if v != 'False':
            cmd.append(f'--{k}')
            cmd.append(f'{v}')

    env = os.environ.copy()
    p = Popen(cmd, env=env)
    p.communicate()


def write_slurm_file(grid, slurm_name, gpu=False):
    sweep_list = list(dict(zip(grid.keys(), values)) 
                    for values in itertools.product(*grid.values()))
    n_jobs = len(sweep_list)

    dir = os.getcwd()
    f = open(os.path.join(dir, f'{slurm_name}.slurm'), 'w')

    f.write('#!/bin/bash\n')
    f.write(f'#SBATCH --job-name={slurm_name}\n')
    f.write('#SBATCH --open-mode=append\n')
    f.write('#SBATCH --output=slurm/%j_%x.out\n')
    f.write('#SBATCH --error=slurm/%j_%x.err\n')
    f.write('#SBATCH --export=ALL\n')
    f.write('#SBATCH --time=10:00:00\n')
    f.write('#SBATCH --mem=32G\n')
    f.write('#SBATCH -c 4\n')
    if gpu:
        f.write('#SBATCH --gres=gpu:1\n')
    
    f.write(f'#SBATCH --array=1-{n_jobs}\n')

    f.write(f'singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "\n source /ext3/env.sh \n python $SCRATCH/jax_continuous_rl/experiments/sweeps/{slurm_name}.py $SLURM_ARRAY_TASK_ID \n" ')

    f.close()