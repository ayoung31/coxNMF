#!/bin/bash

#SBATCH --mem-per-cpu=2g
#SBATCH -n 100
#SBATCH -t 02:00:00
#SBATCH --array=3-4
#SBATCH --output=out/k%a.out

R CMD BATCH run_real.R out/k${SLURM_ARRAY_TASK_ID}.Rout