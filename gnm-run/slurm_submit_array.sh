#!/bin/sh
# time in minutes here.
#SBATCH --time=1
#SBATCH --ntasks=1
#SBATCH --array=0-583

echo `/store/DAMTPEGLEN/mw894/slurm/sample_%03d.dat $SLURM_ARRAY_TASK_ID`
JOB=`printf /store/DAMTPEGLEN/mw894/slurm/sample_%03d.dat $SLURM_ARRAY_TASK_ID`
echo $JOB
/mhome/damtp/r/mw894/diss/gnm-run/cmake-build-debug/slurmRun $JOB