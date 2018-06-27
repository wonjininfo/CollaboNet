import numpy as np
import random
import pickle

def makeTup(Path): 
    sentenceTuple = dict()
    wordset = set()
    wordsetTD = set()
    charset = set()
    charset.add('')
    sentenceTuple['train_dev']=list()
    for filename in Path:
        key = filename[:filename.find('setPath')]
        sentenceTuple[key] = list()
        with open(Path[filename], 'rb') as file_:
            templist = list()
            for line in file_:
                line = line.strip()
                if line != '':
                    temp = line.split('\t')
                    if 'B-' in temp[1]:
                        label = 2
                    elif 'I-' in temp[1]:
                        label = 3
                    elif 'E-' in temp[1]:
                        label = 4
                    elif 'S-' in temp[1]:
                        label = 1
                    else:
                        label = 0
                    templist.append((temp[0],label))
                    wordset.add(temp[0])
                    if key != 'test':
                        wordsetTD.add(temp[0])
                    for c in temp[0]:
                        charset.add(c)
                elif line == '':
                    sentenceTuple[key].append(templist)
                    if key != 'test':
                        sentenceTuple['train_dev'].append(templist)
                    templist = list()
    return sentenceTuple, wordset, wordsetTD, charset

def makeIDTup(name, sentenceTuple, wordset, wordsetTD, charset, word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char):

    idxcounter = len(word2ID)
    idx = 0
    for word in wordset:
        if word not in word2ID:
            word2ID[word] = idxcounter + idx
            ID2word[idxcounter+idx] = word
            idx+=1

    idxcounter = len(word2IDTD)
    idx = 0
    for word in wordsetTD:
        if word not in word2IDTD:
            word2IDTD[word] = idxcounter + idx
            ID2wordTD[idxcounter+idx] = word
            idx+=1
    
    idxcounter = len(char2ID)
    idx = 0
    for char in charset:
        if char not in char2ID:
            char2ID[char] = idxcounter + idx
            ID2char[idxcounter+idx] = char
            idx+=1

    sentenceTupleID = dict()
    sentenceTupleLen = dict()
    for key in sentenceTuple:
        sentenceTupleID[key] = list()
        sentenceTupleLen[key] = list()
        for sent in sentenceTuple[key]:
            sentenceTupleLen[key].append(len(sent))
            tupleID = list()
            for tup in sent:
                tempTup = (word2ID[tup[0]],[char2ID[_] for _ in tup[0]], tup[1])
                tupleID.append(tempTup)
            sentenceTupleID[key].append(tupleID)

    pickle.dump(sentenceTupleID, open('data/'+name+'_sentenceTupleID.pickle','wb'))
    pickle.dump(sentenceTupleLen, open('data/'+name+'_sentenceTupleLen.pickle','wb'))

    return word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char

Path=dict()
word2ID = dict()
ID2word = dict()
word2IDTD = dict()
ID2wordTD = dict()
char2ID = dict()
ID2char = dict()

#NCBI
Path['devsetPath'] =  '../MTL-Bioinformatics-2016/data/NCBI-disease-IOBES/devel.tsv'
Path['testsetPath'] = '../MTL-Bioinformatics-2016/data/NCBI-disease-IOBES/test.tsv'
Path['trainsetPath'] ='../MTL-Bioinformatics-2016/data/NCBI-disease-IOBES/train.tsv'
sentenceTuple, wordset, wordsetTD, charset = makeTup(Path)
word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char = makeIDTup('NCBI',sentenceTuple, wordset, wordsetTD, charset, word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char)
#BC2GM
Path['devsetPath'] =  '../MTL-Bioinformatics-2016/data/BC2GM-IOBES/devel.tsv'
Path['testsetPath'] = '../MTL-Bioinformatics-2016/data/BC2GM-IOBES/test.tsv'
Path['trainsetPath'] ='../MTL-Bioinformatics-2016/data/BC2GM-IOBES/train.tsv'
sentenceTuple, wordset, wordsetTD, charset = makeTup(Path)
word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char = makeIDTup('BC2GM',sentenceTuple, wordset, wordsetTD, charset, word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char)
#BC4CHEMD
Path['devsetPath'] =  '../MTL-Bioinformatics-2016/data/BC4CHEMD-IOBES/devel.tsv'
Path['testsetPath'] = '../MTL-Bioinformatics-2016/data/BC4CHEMD-IOBES/test.tsv'
Path['trainsetPath'] ='../MTL-Bioinformatics-2016/data/BC4CHEMD-IOBES/train.tsv'
sentenceTuple, wordset, wordsetTD, charset = makeTup(Path)
word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char = makeIDTup('BC4CHEMD',sentenceTuple, wordset, wordsetTD, charset, word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char)
#BC5CDR-disease
Path['devsetPath'] =  '../MTL-Bioinformatics-2016/data/BC5CDR-disease-IOBES/devel.tsv'
Path['testsetPath'] = '../MTL-Bioinformatics-2016/data/BC5CDR-disease-IOBES/test.tsv'
Path['trainsetPath'] ='../MTL-Bioinformatics-2016/data/BC5CDR-disease-IOBES/train.tsv'
sentenceTuple, wordset, wordsetTD, charset = makeTup(Path)
word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char = makeIDTup('BC5CDR-disease',sentenceTuple, wordset, wordsetTD, charset, word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char)
#BC5CDR-chem
Path['devsetPath'] =  '../MTL-Bioinformatics-2016/data/BC5CDR-chem-IOBES/devel.tsv'
Path['testsetPath'] = '../MTL-Bioinformatics-2016/data/BC5CDR-chem-IOBES/test.tsv'
Path['trainsetPath'] ='../MTL-Bioinformatics-2016/data/BC5CDR-chem-IOBES/train.tsv'
sentenceTuple, wordset, wordsetTD, charset = makeTup(Path)
word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char = makeIDTup('BC5CDR-chem',sentenceTuple, wordset, wordsetTD, charset, word2ID, ID2word, word2IDTD, ID2wordTD, char2ID, ID2char)

#JNLPBA
Path['testsetPath'] = '../MTL-Bioinformatics-2016/data/JNLPBA/original-data/test/Genia4EReval1.iob2'
Path['trainsetPath'] ='../MTL-Bioinformatics-2016/data/JNLPBA/original-data/train/Genia4ERtask1.iob2'
temp_sentenceTuple = dict()
sentenceTuple = dict()
wordset = set()
wordsetTD = set()
charset = set()
for filename in Path:
    key = filename[:filename.find('setPath')]
    temp_sentenceTuple[key] = list()
    with open(Path[filename], 'rb') as file_:
        templist = list()
        for line in file_:
            line = line.strip()
            if line != '':
                temp = line.split('\t')
                if temp[0] == '':
                    print(line)
                    continue             
                if temp[1] == 'O':
                    tag = 0
                elif ('B-' in temp[1]) or ('b-' in temp[1]):
                    if 'cell_' in temp[1]:
                        tag = 0
                    else:
                        tag = 1
                elif ('I-' in temp[1]) or ('i-' in temp[1]):
                    if 'cell_' in temp[1]:
                        tag = 0
                    else:
                        tag = 2
                templist.append((temp[0],tag))
                wordset.add(temp[0])
                if key != 'test':
                    wordsetTD.add(temp[0])
                for c in temp[0]:
                    charset.add(c)
            elif line == '':
                temp_sentenceTuple[key].append(templist)
                templist = list()
    
    sentenceTuple[key] = list()
    for sent in temp_sentenceTuple[key]:
        temp = list()
        for idx, label in enumerate(sent):
            if idx == len(sent)-1:
                if label[1] == 2 : #L
                    tag = 4 
                else:
                    tag = label[1]
            elif label[1] == 1 and sent[idx+1][1] == 2: #B
                tag = 2
            elif label[1] == 2 and (sent[idx+1][1] == 0 or sent[idx+1][1]== 1): #L
                tag = 4
            elif label[1] == 2 and sent[idx+1][1] == 2: #I
                tag = 3
            else:
                tag = label[1]
                if tag == 2:
                    print(sent, idx)
            temp.append((label[0],tag))
        sentenceTuple[key].append(temp)

idxcounter = len(word2ID)
idx = 0
for word in wordset:
    if word not in word2ID:
        word2ID[word] = idxcounter+idx
        ID2word[idxcounter+idx] = word
        idx+=1

idxcounter = len(word2IDTD)
idx = 0
for word in wordsetTD:
    if word not in word2IDTD:
        word2IDTD[word] = idxcounter+idx
        ID2wordTD[idxcounter+idx] = word
        idx+=1

idxcounter = len(char2ID)
idx = 0
for char in charset:
    if char not in char2ID:
        char2ID[char] = idxcounter+idx
        ID2char[idxcounter+idx] = char
        idx+=1

sentenceTupleID = dict()
sentenceTupleLen = dict()
for key in sentenceTuple:
    sentenceTupleID[key] = list()
    sentenceTupleLen[key] = list()
    for sent in sentenceTuple[key]:
        sentenceTupleLen[key].append(len(sent))
        tupleID = list()
        for tup in sent:
            charlist = list()
            tempTup = (word2ID[tup[0]],[char2ID[_] for _ in tup[0]], tup[1])
            tupleID.append(tempTup)
        sentenceTupleID[key].append(tupleID)

dev_idx = random.sample(range(0,len(sentenceTupleLen['train'])),len(sentenceTupleLen['test']))

sentenceTupleID['dev'] = list()
sentenceTupleLen['dev'] = list()
for idx in dev_idx:
    sentenceTupleID['dev'].append(sentenceTupleID['train'][idx])
    sentenceTupleLen['dev'].append(sentenceTupleLen['train'][idx])
    
tempID = list()
tempLen = list()
for i in range(len(sentenceTupleLen['train'])):
    if i in dev_idx:
        pass
    else:
        tempID.append(sentenceTupleID['train'][i])
        tempLen.append(sentenceTupleLen['train'][i])

sentenceTupleLen['train_dev'] = sentenceTupleLen['train'][:]
sentenceTupleID['train_dev'] = sentenceTupleID['train'][:]

sentenceTupleLen['train'] = tempLen[:]
sentenceTupleID['train'] = tempID[:]

pickle.dump(sentenceTupleID, open('data/JNLPBA_sentenceTupleID.pickle','wb'))
pickle.dump(sentenceTupleLen, open('data/JNLPBA_sentenceTupleLen.pickle','wb'))

word2ID['NUM'] = 107551
ID2word[107551] = 'NUM'
word2ID['<UNK>'] = 107552
ID2word[107552] = '<UNK>'

pickle.dump(char2ID, open('data/char2ID.pickle','wb'))
pickle.dump(word2ID, open('data/word2ID.pickle','wb'))
pickle.dump(ID2char, open('data/ID2char.pickle','wb'))
pickle.dump(ID2word, open('data/ID2word.pickle','wb'))
pickle.dump(word2IDTD, open('data/word2IDTD.pickle','wb'))
pickle.dump(ID2wordTD, open('data/ID2wordTD.pickle','wb'))
