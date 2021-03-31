#!/bin/sh

OUT_DIR=result.LR_`echo $LR`.MAX_TOKENS_`echo $MAX_TOKENS`
LR=0.1
MAX_TOKENS=4000
mkdir ${OUT_DIR}

#SBATCH --job-name=akkadian
#SBATCH --output=${OUT_DIR}/out
#SBATCH --error=${OUT_DIR}/err
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4

./akkadian_fairseq.sh ${OUT_DIR} ${LR} ${MAX_TOKENS}
