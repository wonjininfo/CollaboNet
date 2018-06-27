import numpy as np 
import tensorflow as tf

def isNum(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def wiki_wordVecProcessing(word2ID, word2IDTD):
    wordVecFilePath="wordvec/pubmed_pmc_wiki_200dim_wordvec.txt"
    wordVec2LineNo, wordEmbedding = wiki_loadWordVec(wordVecFilePath)
    ID2wordVecIdx=dict()
    i = 1
    for key in word2ID:
        if wordVec2LineNo.has_key(key):
            ID2wordVecIdx[word2ID[key]]=wordVec2LineNo[key] # inp : word ID -> word Line No
        else: #UNK
            if isNum(key):
                ID2wordVecIdx[word2ID[key]]=wordVec2LineNo['NUM']
            else:
                ID2wordVecIdx[word2ID[key]]=98734 #<UNK>
    return ID2wordVecIdx, wordVec2LineNo, wordEmbedding

def wiki_loadWordVec(path,vocaPath="wordvec/vocab.txt"):
    wordVec2LineNo=dict()
    vectList=list()
    
    vectList.append(np.array([0]*200)) #pad 
    with open(path,'rb') as vecFile:
        for line in vecFile:
            vecValRow=line.split()
            vect=map(float, vecValRow)
            vectList.append(vect)
    vectList.append(np.random.rand(200))#UNK
    wordVecArrayTmp=np.asarray(vectList,dtype='float32')
    
    with open(vocaPath,'rb') as vocaFile:
        for itera, line in enumerate(vocaFile):
            wordVec2LineNo[line.strip('\n')]=itera+1 #pad
    
    wordVec2LineNo['<UNK>']=itera+2
    wordEmbedding=wordVecArrayTmp
    embedding_dim=len(wordVecArrayTmp[0])
    
    return wordVec2LineNo,wordEmbedding

def char_padding(inputs, voca_size, embedding_dim, wordMaxLen, charMaxLen):
    sentences_embed=list()
    sentences_embed_len=list()
    
    for senIdx,sentence in enumerate(inputs):
        inputs_embed=list()
        inputs_embed_len=list()
        
        for wordIdx, words in enumerate(sentence): #one sentence = list of words
            words_padded = [0] + words + [0]*(charMaxLen-(1 + len(words)))
            inputs_embed.append(words_padded)
            inputs_embed_len.append(len(words))
            
        paddings=[0]*charMaxLen
        inputs_embed=inputs_embed+[paddings]*(wordMaxLen - len(inputs_embed))
        sentences_embed.append(inputs_embed)
        sentences_embed_len.append(inputs_embed_len)
        
    return sentences_embed,sentences_embed_len

def embedding_lookup(inputs, voca_size, initializer, reuse=False, trainable=True, scope='Embedding'):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        embedding_tablePAD = tf.get_variable("embedPAD",
                        initializer=initializer[0:1], trainable=False, dtype=tf.float32)
        embedding_tableLast = tf.get_variable("embedLast",
                        initializer=initializer[1:], trainable=trainable, dtype=tf.float32)
        embedding_table=tf.concat([embedding_tablePAD,embedding_tableLast],axis=0,name="embed")
        inputs_embed = tf.nn.embedding_lookup(embedding_table, inputs) 
        
        return inputs_embed, embedding_table

def char_embedding(inputs, voca_size, embedding_dim, length, charMaxLen, initializer=None, reuse=False, trainable=True, scope='EmbeddingChar'):
    #initializer : wordvec, max length=max Length of char vec
    
    if initializer == None:
        initializer=np.concatenate((np.zeros((1,embedding_dim),dtype='float32'), #Padding : [1,dim]
                                    np.random.rand(voca_size-1,embedding_dim).astype('float32')), axis=0) #Other than Padding [vocasize-1,dim]
    else:
        initializer=np.concatenate((np.zeros((1,embedding_dim),dtype='float32'), #Padding [1,dim]
                                    initializer), axis=0) #Other than Padding [vocasize-1,dim]
        
    with tf.variable_scope(scope, reuse=reuse) as scope:
        embedCharPAD = tf.get_variable("embedCharPad",
                                       initializer=initializer[0:1], trainable=False, dtype=tf.float32)
        embedCharLast = tf.get_variable("embedCharLast",
                                initializer=initializer[1:], trainable=trainable, dtype=tf.float32)
        embedChar=tf.concat([embedCharPAD,embedCharLast],axis=0,name="embedChar")
        embedded = tf.nn.embedding_lookup(embedChar, inputs)

        return embedded
