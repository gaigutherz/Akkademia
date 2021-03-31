#!/bin/sh
#SBATCH --job-name=akkadian_fairseq
#SBATCH --output=akkadian_fairseq.out
#SBATCH --error=akkadian_fairseq.err
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
./akkadian_fairseq.py

