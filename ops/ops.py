import os
import errno
import time
import random
import numpy as np
import csv

def setExpName(savePath='modelSave/'):
    timeNow=time.strftime("%Y%m%d%H%M")
    timeNow_edited=str(timeNow)
    print(timeNow_edited)

    suffix=0
    while 1: # making foldr code : if same name exists -> add [_num]
        if suffix>10: # for error treating 
            break
            
        try:
            if suffix==0:
                os.mkdir(savePath+timeNow_edited)
            else:
                os.mkdir(savePath+timeNow_edited+'_'+str(suffix))
                timeNow_edited=timeNow_edited+'_'+str(suffix)
            break
        except OSError as e:
            if e.errno == errno.EEXIST: # if file exists! Python2.7 doesn't support file exist exception so need to use this
                print(timeNow_edited+'_'+str(suffix)+' Directory exists! not created.')
                suffix+=1
            else:
                raise
            
    return timeNow_edited

def prf(preds, answers, length):
    try:
        return prf_lu(preds, answers, length)
    except:
        return 0,0,0

def prf_lu(preds, answers, length):
    answer = 0
    found = 0
    match = 0
    PMIDS = list()
    
    for idxSent, sent in enumerate(preds):
        for idxWord, word in enumerate(sent):
            if word == 2 and idxWord<length[idxSent]:
                found += 1
            if word == 1 and idxWord<length[idxSent]:
                found += 1
    
    for idxSent, sent in enumerate(answers):
        for idxWord, word in enumerate(sent):
            if word == 2 and idxWord<length[idxSent]:
                answer += 1
            if word == 1 and idxWord<length[idxSent]:
                answer += 1

    for idxSent, sent in enumerate(answers):

        answSentStr = ''.join(map(str,sent))
        predSentStr = ''.join(map(str,preds[idxSent]))
        
        seqOp = 0
        while True: # find U
            
            op_u = answSentStr[seqOp:].find('1')
            
            if op_u == -1: 
                break
                
            seqOp += op_u
            
            if predSentStr[seqOp]=='1':
                PMIDS.append(idxSent)
                match += 1
                
            seqOp += 1
            
            
        seqOp = 0
        while True: # find B I L
            op_b = answSentStr[seqOp:].find('2')
            
            if op_b == -1:
                break
            seqOp += op_b
            
            if predSentStr[seqOp]=='2':
                i = 1
                while True:
                    chkWord = predSentStr[seqOp+i]
                    if not answSentStr[seqOp+i] == chkWord :
                        break
                    elif chkWord == '4':
                        PMIDS.append(idxSent)
                        match += 1
                        break
                    elif chkWord == '3':
                        i += 1 
                        
            seqOp += 1

    if found!=0:
        precision=match/float(found)
    else:
        precision=0
    if answer!=0:
        recall=match/float(answer)
    else:
        recall=0
    if precision!=0 and answer!=0:
        f1score=2*precision*recall/(precision+recall)
    else:
        f1score=0
    
    return precision, recall, f1score
        
def dataSplitdoc(PMID2IDtuple, PMID2IDtupleLen, PMIDListDict, ID2wordVecIdx):
    answerData = dict()
    lengthData = dict()
    x_data = dict()
    x_char_data = dict()
    
    for key in PMIDListDict:
        answerData[key] = list()
        lengthData[key] = list()
        x_data[key] = list()
        x_char_data[key] = list()
        for PMID in PMIDListDict[key]:
            answerData[key].append([_[2] for _ in PMID2IDtuple[PMID]])
            lengthData[key].append(PMID2IDtupleLen[PMID])
            x_data[key].append([ID2wordVecIdx[_[0]] for _ in PMID2IDtuple[PMID]])
            x_char_data[key].append([_[1] for _ in PMID2IDtuple[PMID]])
            
    return x_data, x_char_data, answerData, lengthData

def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_oneHot(input_list, depth, length): #input_list [batch_size,sequence_size]
    oneHot_output = list()
    for i, sent in enumerate(input_list):
        temp = np.zeros((length[i], depth))
        tempsent = sent[:length[i]]
        temp[np.arange(length[i]), tempsent] = 1
        oneHot_output.append(temp)
    return oneHot_output

def viterbi_pp(predictionResult, lengthData, num_classes=5):
    if num_classes == 5:
        for docIdx, doc in enumerate(predictionResult):
            prev = 0
            for predIdx, pred in enumerate(doc):
                false_I = False

                if prev == 0:
                    if pred == 4 or pred == 3:
                        predictionResult[docIdx][predIdx] = 0

                elif prev == 1:
                    if pred == 4 or pred == 3:
                        predictionResult[docIdx][predIdx] = 0

                elif prev == 2:
                    if pred == 0 or pred == 2 or pred == 1:
                        predictionResult[docIdx][predIdx-1] = 0

                elif prev == 4:
                    if pred == 3 or pred == 4:
                        predictionResult[docIdx][predIdx] = 0

                if pred == 2 and predIdx < lengthData[docIdx]-2:
                    for tidx, t in enumerate(doc[predIdx+1:]):
                        if t == 4: 
                            break
                        elif t != 3:
                            false_I = True
                            false_flag = tidx
                            break
                    if false_I:
                        for i in range(false_flag):
                            predictionResult[docIdx][predIdx+i] = 0
                        
                elif pred == 2 and predIdx == lengthData[docIdx]-1:
                    predictionResult[docIdx][predIdx] = 0
                    prev = 0

                if predIdx == lengthData[docIdx]-1:
                    break

                prev = predictionResult[docIdx][predIdx]
            predictionResult[docIdx] = predictionResult[docIdx][:lengthData[docIdx]]
        return predictionResult

    elif num_classes == 3:
        for docIdx, doc in enumerate(predictionResult):
            prev = 0
            for predIdx, pred in enumerate(doc):
                if prev == 0:
                    if pred == 2:
                        predictionResult[docIdx][predIdx] = 0
                prev = predictionResult[docIdx][predIdx]
                if preIdx == lengthData[docIdx]-1:
                    break
            predictionResult[docIdx] = predictionResult[docIdx][:lengthData[docIdx]]
        return predictionResult
    
    else:
        print('num class error, only 3:bio 5:biolu')
        return 0
