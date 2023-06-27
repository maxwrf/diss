#!/bin/sh
# time in minutes here.
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --array=0-586

JOB=`printf /Users/maxwuerfek/code/diss/gnm-run/slurm/sample_%03d.dat $SLURM_ARRAY_TASK_ID`
echo $JOB
/Users/maxwuerfek/code/diss/gnm-run/cmake-build-debug/slurmRun $JOB