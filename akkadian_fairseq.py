#!/bin/sh

../fairseq/fairseq_cli/preprocess.py --source-lang ak --target-lang en --trainpref NMT_input/tokenization/train --validpref NMT_input/tokenization/valid --testpref NMT_input/tokenization/test --destdir data-bin --thresholdtgt 0 --thresholdsrc 0 --workers 10

mkdir -p data-bin/checkpoints

../fairseq/fairseq_cli/train.py data-bin --arch fconv --dropout 0.1 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --clip-norm 0.1 --lr 0.05 --lr-scheduler fixed --force-anneal 50 --max-tokens 2000 --no-epoch-checkpoints --patience 3 --save-dir data-bin/checkpoints

#../fairseq/fairseq_cli/generate.py data-bin-lr0.05 --path data-bin-lr0.05/checkpoint_best.pt --beam 5
