#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -J 1-30

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate scVIpy311

cd $PBS_O_WORKDIR

python single-repeat-run.py --repeat $PBS_ARRAY_INDEX