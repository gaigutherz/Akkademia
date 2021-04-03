#!/bin/sh

run() {
	LR=$1
	MAX_TOKENS=$2
	OUT_DIR=result.LR_${LR}.MAX_TOKENS_${MAX_TOKENS}
	mkdir -p ${OUT_DIR}
	cp akkadian_fairseq.template ${OUT_DIR}/akkadian_fairseq.slurm
	sed -i "s/LR_PLACEHOLDER/${LR}/g" ${OUT_DIR}/akkadian_fairseq.slurm
	sed -i "s/MAX_TOKENS_PLACEHOLDER/${MAX_TOKENS}/g" ${OUT_DIR}/akkadian_fairseq.slurm
	cd ${OUT_DIR}
	sbatch akkadian_fairseq.slurm
	cd ..
}

for LR in 0.05 0.1
do
	for MAX_TOKENS in 4000 8000
	do
		run ${LR} ${MAX_TOKENS}
	done
done

