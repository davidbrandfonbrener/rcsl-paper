#!/bin/bash
#SBATCH --job-name=joint_rvs_sweep
#SBATCH --open-mode=append
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=1-90
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
 source /ext3/env.sh 
 python $SCRATCH/jax_continuous_rl/experiments/sweeps/joint_rvs_sweep.py $SLURM_ARRAY_TASK_ID 
" 