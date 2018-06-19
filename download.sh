#!/bin/bash

# Create data folder and download preprocessed file
mkdir data
cd data
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC2GM_sentenceTupleID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC2GM_sentenceTupleLen.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC4CHEMD_sentenceTupleID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC4CHEMD_sentenceTupleLen.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC5CDR-chem_sentenceTupleID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC5CDR-chem_sentenceTupleLen.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC5CDR-disease_sentenceTupleID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/BC5CDR-disease_sentenceTupleLen.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/JNLPBA_sentenceTupleID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/JNLPBA_sentenceTupleLen.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/NCBI_sentenceTupleID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/NCBI_sentenceTupleLen.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/word2ID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/word2IDTD.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/char2ID.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/ID2word.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/ID2wordTD.pickle
wget https://s3-us-west-2.amazonaws.com/collabonet/data/ID2char.pickle

cd ..
mkdir wordvec
cd wordvec
wget https://s3-us-west-2.amazonaws.com/collabonet/wordvec/vocab.txt
wget https://s3-us-west-2.amazonaws.com/collabonet/wordvec/pubmed_pmc_wiki_200dim_wordvec.txt

cd ..
mkdir modelSave
