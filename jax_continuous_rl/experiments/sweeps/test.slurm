#!/bin/bash
#SBATCH --job-name=test
#SBATCH --open-mode=append
#SBATCH --output=$SCRATCH/jax_continuous_rl/experiments/sweeps/slurm/%j_%x.out
#SBATCH --error=$SCRATCH/jax_continuous_rl/experiments/sweeps/slurm/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

#SBATCH --array=1-4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
python $SCRATCH/jax_continuous_rl/experiments/sweeps/test.py $SLURM_ARRAY_TASK_ID
"