#!/bin/bash

#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

#SBATCH --job-name=rkhs
#SBATCH --output=%x-%A-%3a.out
#SBATCH --error=%x-%A-%3a.err
#SBATCH --array=1-100
##SBATCH --constraint=a100

module load miniconda/22.11.1-1
module load cuda/11.4.0
module load cudnn/cuda11-8.4.1.50

conda activate rkhs_gpu

out=$(sed -n ${SLURM_ARRAY_TASK_ID}p sim_args.txt)
echo $out
python python/gpytorch_bake.py $out
