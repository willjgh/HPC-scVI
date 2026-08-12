#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=1:mem=4gb

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate scVIpy311

cd $PBS_O_WORKDIR

python batch-run-cleanup.py
bash clean-logs
