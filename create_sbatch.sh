#!/bin/bash

counter=0
for w in 35 40 45;
do
	for n in 5 10 15 10 30;
	do
		for t in 3 5 10;
		do
			echo "#!/bin/bash
#SBATCH -J ASHP
#SBATCH --output=/p/project/cvsk18/lammert1/ASHP/job_w${w}_n${n}_t${t}.log
#SBATCH --error=/p/project/cvsk18/lammert1/ASHP/job_w${w}_n${n}_t${t}.err
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --account=vsk18
#SBATCH --cpus-per-task=128
#SBATCH --ntasks=1

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

srun --cpu-bind=none python3 ASHP.py -i comb.mrcs -c 5 -a Spectral_clustering -o /p/project/cvsk18/lammert1/ASHP/output/run_MSE_w${w}_n${n}_t${t} -m MSE -w ${w} -n ${n} -t ${t} -s 60 
srun --cpu-bind=none python3 ASHP.py -i comb.mrcs -c 5 -a Spectral_clustering -o /p/project/cvsk18/lammert1/ASHP/output/run_MSE_l_w${w}_n${n}_t${t} -m MSE -w ${w} -n ${n} -t ${t} -s 60 -l
srun --cpu-bind=none python3 ASHP.py -i comb.mrcs -c 5 -a Spectral_clustering -o /p/project/cvsk18/lammert1/ASHP/output/run_conv_w${w}_n${n}_t${t} -m conv -w ${w} -n ${n} -t ${t} -s 60 -f
srun --cpu-bind=none python3 ASHP.py -i comb.mrcs -c 5 -a Spectral_clustering -o /p/project/cvsk18/lammert1/ASHP/output/run_conv_l_w${w}_n${n}_t${t} -m conv -w ${w} -n ${n} -t ${t} -s 60 -l -f

" > "sbatch_w${w}_n${n}_t${t}.sh"


		done
	done
done
