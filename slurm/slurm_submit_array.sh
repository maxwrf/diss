#!/bin/sh
# time in minutes here.
#SBATCH --time=5
#SBATCH --ntasks=1
#SBATCH --array=0-249

JOB=`printf sample_%03d.dat.npz $SLURM_ARRAY_TASK_ID`
echo $JOB
python3 slurm_run_gnms.py $JOB