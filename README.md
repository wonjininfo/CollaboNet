# Collaborating different entity types for biomedical Named Entity Recognition

This project provides a neural network(bi-LSTM + CRF) based biomedical Named Entity Recognition.  
Our implementation is based on the Tensorflow library in python.  
  
* __TITLE__  :  Collaborating different entity types for biomedical Named Entity Recognition
* __AUTHOR__ :  Wonjin Yoon<sup>1!</sup>, Chan Ho So<sup>2!</sup>, Jinhyuk Lee<sup>1</sup> and Jaewoo Kang<sup>1\*</sup>
    * __Author details__  
    <sup>\*</sup> Correspondence : kangj@korea.ac.kr  
    <sup>1</sup> Department of Computer Science and Engineering, Korea University  
    <sup>2</sup> Interdisciplinary Graduate Program in Bioinformatics, Korea University  
    <sup>!</sup> Equal contributor  


## Quick Links

- [Requirements](#requirements)
- [Model](#model)
- [Data](#data)
- [Usage](#usage)
- [Performance](#performance)

## Requirements
One GPU device is required for execution of this project codes.  
python 2.7  
numpy 1.14.2  
tensorflow-gpu 1.7.0  

### License
MIT License would be fine but it seems like we need to look a little bit more on licenses for other libraries which we used.  
Please refer <a href=./License-thirdparty.txt>License-thirdparty.txt</a> file  

## Model
**[LEFT]** Character level word embedding using CNN and overview of Bidirectional LSTM with Conditional Random Field (BiLSTM-CRF).  
**[RIGHT]** Structure of CollaboNet when Gene model act as a role of target model. Rhombus represents the CRF layer. Arrows show the flow of information when target model is training. Dashed arrows mean that information is not flowing when target model is under training.  
![Model](https://s3-us-west-2.amazonaws.com/collabonet/model_tot.jpg)

## Data
### Train, Test Data
We used the datasets collected by Crichton et al. \cite{crichton2017neural} for the experiment.  
The datasets can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).  
We found the JNLPBA dataset from Crichton et al. has minor mistake on sentence separation.  
So we re-generated the dataset from the original data by [Kim et al.](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/shared_task_intro.pdf)   

The details of each dataset are showed below:  


|               Corpora               |  Entity type  | No. sentence | No. annotations |     Data Size    |
|:-----------------------------------:|:-------------:|:------------:|:---------------:|:----------------:|
|  NCBI-Disease (Dogan et al., 2014)  |    Disease    |     7,639    |      6,881      |   793 abstracts  |
|      JNLPBA (Kim et al., 2004)      | Gene/Proteins |    22,562    |      35,336     |  2,404 abstracts |
|       BC5CDR (Li et al., 2016)      |   Chemicals   |    14,228    |      15,935     |  1,500 articles  |
|       BC5CDR (Li et al., 2016)      |    Diseases   |    14,228    |      12,852     |  1,500 articles  |
| BC4CHEMD (Krallinger et al., 2015a) |   Chemicals   |    86,679    |      84,310     | 10,000 abstracts |
|     BC2GM (Akhondi et al., 2014)    | Gene/Proteins |    20,510    |      24,583     | 20,000 sentences |

We preprocess datasets and made them into tsv files. 
The tsv files are publicly available in [download.sh](./download.sh), and we recommend downloading the datasets to run our code.  

### Pre-trained Embedding Data
We used pre-trained word embeddings from [Pyysalo et al.](http://bio.nlplab.org/) which is trained on PubMed, PubMed Central(PMC) and Wikipedia text. Because the embedding file is too large, we used shrinked dataset available in [download.sh](./download.sh).  

## Usage
### Download Data
```
bash download.sh
```

### Single Task Model [STM] (6 datasets)
__Phase 0, Preperation phase__  
```
python run.py --ncbi --jnlpba --bc5_chem --bc5_disease --bc4 --bc2 --epoch 50 --lr_pump --lr_decay 0.05
```
You can also refer to [stm.sh](./stm.sh) for detailed usage.

### CollaboNet (6 datasets)
__You should make pre-trained model with STM before running CollaboNet.__  
```
python run.py --ncbi --jnlpba --bc5_chem --bc5_disease --bc4 --bc2 --epoch 30 --lr_pump --lr_decay 0.05 --pretrained STM_MODEL_DIRECTORY_NAME(ex 201806210605)
```
You can also refer to [collabo.sh](./collabo.sh) for detailed usage.


## Performance
### STM
|           Model          |          | NCBI-disease | JNLPBA | BC5CDR-chem | BC5CDR-disease | BC4CHEMD | BC2GM | Average |
|:------------------------:|:--------:|:------------:|:------:|:-----------:|:--------------:|:--------:|:-----:|:-------:|
| Habibi et al. (2017) STM | F1 Score |     84.44    |  77.25 |    90.63    |      83.49     |   86.62  | 77.82 |  83.38  |
|  Wang et al. (2018) STM  | F1 Score |     83.92    |  72.17 |    *89.85   |     *82.68     |   **88.75**  | **80.00** |  82.90  |
|          **Our STM**         | F1 Score |     **85.19**    |  **77.77** |    **92.79**    |      **83.54**     |   88.40  | 79.27 |  **84.49**  |
* Scores in the asterisked (\*) cells are re-experimented performances by us, as the scores are not reported in the original papers.   
* The best scores in these experiments are in bold.  

### CollaboNet
|                        |          | NCBI-disease |   JNLPBA  | BC5CDR-chem | BC5CDR-disease |  BC4CHEMD |   BC2GM   | Average |
|:----------------------:|:--------:|:------------:|:---------:|:-----------:|:--------------:|:---------:|:---------:|:-------:|
| Wang et al. (2018) MTM | F1 Score |     86.14    |   73.52   |    *91.29   |     *83.33     | **89.37** | **80.74** |  84.07  |
|   **Our CollaboNet**   | F1 Score |   **86.36**  | **78.58** |  **93.31**  |    **84.08**   |   88.85   |   79.73   |  **85.15**  |
* Scores in the asterisked (\*) cells are re-experimented performances by us, as the scores are not reported in the original papers.   
* The best scores in these experiments are in bold.  
