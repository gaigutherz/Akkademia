sudo apt-get install python3.7
sudo apt-get install python3-pip
pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
sudo apt-get install git
git clone https://github.com/pytorch/fairseq
cd fairseq
sudo pip3 install --editable ./
cd ..
git clone https://github.com/pytorch/fairseq

git clone https://github.com/gaigutherz/Akkademia.git
cd Akkademia

# preprocess
TEXT=NMT_input
fairseq-preprocess \
    --source-lang ak --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin --thresholdtgt 0 --thresholdsrc 0 --workers 60