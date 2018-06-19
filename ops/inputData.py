import pickle
import random
import numpy as np
from ops import *
from embeddingOps import *

def input_datapickle(name, ID2wordVecIdx, TF_result=False):
    sentenceTupleID = pickle.load(open('data/'+name+'_sentenceTupleID.pickle','rb'))
    sentenceTupleLen = pickle.load(open('data/'+name+'_sentenceTupleLen.pickle','rb'))
    
    x_data = dict()
    x_char_data = dict()
    lengthData = dict()
    answerData = dict()
    TF_dict = {0:0, 1:1, 2:1, 3:1, 4:1}
    for key in sentenceTupleID:
        x_data[key] = list()
        x_char_data[key] = list()
        lengthData[key] = list()
        answerData[key] = list()
        for idx, sent in enumerate(sentenceTupleID[key]):
            x_data[key].append([ID2wordVecIdx[_[0]] for _ in sent ])
            x_char_data[key].append([_[1] for _ in sent])
            lengthData[key].append(sentenceTupleLen[key][idx])
            if TF_result:
                answerData[key].append([TF_dict[_[2]] for _ in sent])
            else:
                answerData[key].append([_[2] for _ in sent])
    
    return x_data, x_char_data, answerData, lengthData

def input_wordVec():
    word2ID=pickle.load(open('data/word2ID.pickle','rb'))
    word2IDTD=pickle.load(open('data/word2IDTD.pickle','rb'))
    ID2wordVecIdx, wordVec2LineNo, wordEmbedding = wiki_wordVecProcessing(word2ID, word2IDTD)
    
    return ID2wordVecIdx, wordVec2LineNo, wordEmbedding

def shuffle_data(x_data, x_char_data, answerData, lengthData):
    
    s_x_data, s_x_char_data, s_answerData, s_lengthData = zip(
                        *random.sample(list(zip(x_data, x_char_data, answerData, lengthData)), len(x_data)))

    return s_x_data, s_x_char_data, s_answerData, s_lengthData

def batch_sort(lengthData, batch_size):
    batchgroup = dict()
     
    for key in lengthData:
        sort_index = np.argsort(lengthData[key])
        batchgroup[key] = list()

        for count, idx in enumerate(sort_index):
            if (count)%batch_size == 0:
                temp = list()
            temp.append(idx)
            
            if ((count+1)%batch_size == 0) and (count != 0):
                batchgroup[key].append(temp)
            
            elif ((count+1)%batch_size != 0) and ((count+1) == len(sort_index)):
                batchgroup[key].append(temp)
    
    return batchgroup

def idx2data(batchgroup, x_data, x_char_data, answerData, lengthData):
    x_minibatch = list()
    y_minibatch = list()
    xlen_minibatch = list()
    x_char_minibatch = list()
    for idx in batchgroup:
        x_minibatch.append(x_data[idx][:])
        y_minibatch.append(answerData[idx][:])
        x_char_minibatch.append(x_char_data[idx][:])
        xlen_minibatch.append(lengthData[idx])
    
    return x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch

def batch_padding(x_data, answerData, lengthData, x_char_data):
    maxlen = 0
    x_datapad = x_data[:]
    answerDatapad = answerData[:]
    x_char_datapad = x_char_data[:]

    for length in lengthData:
        if length > maxlen:
            maxlen = length
    
    for idx in range(len(x_datapad)):
        for i in range(lengthData[idx], maxlen):
            x_datapad[idx].append(0)
            x_char_datapad[idx].append([0])
            answerDatapad[idx].append(0)
    
    return x_datapad, answerDatapad, x_char_datapad, maxlen
