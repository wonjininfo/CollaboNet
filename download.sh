#!/bin/bash

# Create data folder and download preprocessed file
mkdir data
git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016.git ../MTL-Bioinformatics-2016
python preprocessing.py

mkdir wordvec
cd wordvec
wget https://s3-us-west-2.amazonaws.com/collabonet/wordvec/vocab.txt
wget https://s3-us-west-2.amazonaws.com/collabonet/wordvec/pubmed_pmc_wiki_200dim_wordvec.txt
cd ..

mkdir modelSave
