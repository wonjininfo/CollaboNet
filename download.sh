#!/bin/bash

# Create data folder and download preprocessed file
mkdir data
git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016.git ../MTL-Bioinformatics-2016
python preprocessing.py

mkdir wordvec
cd wordvec
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Q2TCwPRrEPtI75IHJ9awbKI5FH3lJS0w' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Q2TCwPRrEPtI75IHJ9awbKI5FH3lJS0w" -O pubmed_pmc_wiki_200dim_wordvec.txt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1H1VUlvjeGjVGANqoadbn3Fx015JNE_nM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1H1VUlvjeGjVGANqoadbn3Fx015JNE_nM" -O vocab.txt && rm -rf /tmp/cookies.txt
cd ..

mkdir modelSave
