#!/bin/bash
#SBATCH -J ASHP_analyse
#SBATCH --output=analyse_data_stack.log
#SBATCH --error=analyse_data_stack.err
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --account=vsk18
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --cpu-bind=none bash analyse_data_stack_2.sh /p/project/cvsk18/lammert1/ASHP/output/ out-file.dat


