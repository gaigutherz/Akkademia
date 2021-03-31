#!/usr/bin/env bash

sudo apt-get install python3.7
sudo apt-get install python3-pip
pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
sudo apt-get install git
git clone https://github.com/pytorch/fairseq
cd fairseq
sudo pip3 install --editable ./
cd ..

git clone https://github.com/gaigutherz/Akkademia.git
cd Akkademia

# preprocess
../fairseq/fairseq_cli/preprocess.py \
    --source-lang ak --target-lang en --trainpref NMT_input/tokenization/train \
    --validpref NMT_input/tokenization/valid --testpref NMT_input/tokenization/test \
    --destdir data-bin --thresholdtgt 0 --thresholdsrc 0 --workers 10

# train
# Trying learning rate of 0.1 instead of 0.5, trying to solve problem of gradients getting to inf/nan
mkdir -p checkpoints/fconv_ak_en
fairseq-train \
    data-bin \
    --arch fconv \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.1 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir checkpoints/fconv_ak_en

# translate
fairseq-generate \
    data-bin \
    --path checkpoints/fconv_ak_en/checkpoint_best.pt \
    --beam 5 --remove bpe