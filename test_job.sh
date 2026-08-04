#!/bin/bash
#PBS -lwalltime=00:10:00
#PBS -lselect=1:ncpus=1:mem=1gb

ml tools/prod
ml SciPy-bundle/2025.07-iimkl-2025b
source $HOME/venv/scVI-venv/bin/activate

cd $PBS_O_WORKDIR

python3 test-script.py