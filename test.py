import time
import random
import pickle
import sys
import os
import argparse
from copy import deepcopy
import numpy as np
from collections import OrderedDict # for OrderedDict()

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.tensorboard.plugins import projector

from ops.ops import *
from ops.embeddingOps import *
from ops.inputData import *
from model.CollaboNet import *
from model.RunModel import *

if __name__ == '__main__':

    #Hyper Params
    parser = argparse.ArgumentParser()
    parser.add_argument('--guidee_data', type=str, help='data name', default='name')
    parser.add_argument('--pretrained', type=int, help='pretrained expName', default=0)
    parser.add_argument('--ncbi', action='store_true', help='include ncbi data', default=False)
    parser.add_argument('--jnlpba', action='store_true', help='include jnlpba data', default=False)
    parser.add_argument('--bc2', action='store_true', help='include bc2gm data', default=False)
    parser.add_argument('--bc4', action='store_true', help='include bc4chemd data', default=False)
    parser.add_argument('--bc5_disease', action='store_true', help='include bc5-disease data', default=False)
    parser.add_argument('--bc5_chem', action='store_true', help='include bc5-chem data', default=False)
    parser.add_argument('--bc5', action='store_true', help='include bc5cdr data', default=False)
    parser.add_argument('--tensorboard', action='store_true', help='single flag [default]False', default=False)
    parser.add_argument('--epoch', type=int, help='max epoch', default=500)
    parser.add_argument('--num_class', type=int, help='result class bio(3) [default]biolu(5)', default=5)
    parser.add_argument('--ce_dim', type=int, help='char embedding dim', default=30)
    parser.add_argument('--clwe_dim', type=int, help='char level word embedding dim', default=200)
    parser.add_argument('--clwe_method', type=str, help='clwe method: CNN biLSTM', default='CNN')
    parser.add_argument('--batch_size', type=int, help='batch size', default=10)
    parser.add_argument('--hidden_size', type=int, help='lstm hidden layer size', default=300)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--lr_decay', type=float, help='learning rate dacay rate', default=0)
    parser.add_argument('--lr_pump', action='store_true', help='do lr_pump', default=False)
    parser.add_argument('--loss_weight', type=float, help='loss weight between CRF, LSTM', default=1)
    parser.add_argument('--fc_method', type=str, help='fc method', default='normal')
    parser.add_argument('--mlp_layer', type=int, help='num highway layer ', default=1)
    parser.add_argument('--char_maxlen', type=int, help='char max length', default=49)
    parser.add_argument('--embdropout', type=float, help='input embedding dropout_rate', default=0.5)
    parser.add_argument('--lstmdropout', type=float, help='lstm output dropout_rate', default=0.3)
    parser.add_argument('--seed', type=int, help='seed value', default=0)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #tf verbose off(info, warning)
   
    #seed initialize 
    expName = str(args.seed)
    if args.seed != 0:
        seedV = int(args.seed%100000)
    else:
        try:
            tempSeed = int(expName)
        except:
            tempSeed = int(expName[:12])        
        seedV = int(tempSeed%100000)
    
    random.seed(seedV)
    np.random.seed(seedV)
    tf.set_random_seed(seedV)
    
    #gpu setting
    gpu_config = tf.ConfigProto(device_count={'GPU':1})  # only use GPU no.1
    gpu_config.gpu_options.allow_growth = True # only use required resource(memory)
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 1 # restrict to 100%
    
    ID2wordVecIdx, wordVec2LineNo, wordEmbedding = input_wordVec()
    ID2char=pickle.load(open('data/ID2char.pickle','rb'))
    
    m_train = 'train'
    m_dev = 'dev'
    m_test = 'test'

    modelDict=OrderedDict() 
    if args.ncbi: 
        ncbi_args = deepcopy(args)
        ncbi_args.guidee_data = 'NCBI'  
        modelDict['NCBI']={'args':ncbi_args}
    if args.jnlpba:
        jnl_args=deepcopy(args) 
        jnl_args.guidee_data='JNLPBA'
        modelDict['JNLPBA']={'args':jnl_args}
    if args.bc2:
        bc2_args=deepcopy(args) 
        bc2_args.guidee_data='BC2GM'
        modelDict['BC2GM']={'args':bc2_args}
    if args.bc4:
        bc4_args=deepcopy(args)
        bc4_args.guidee_data='BC4CHEMD'
        modelDict['BC4CHEMD']={'args':bc4_args}
    if args.bc5_chem:
        bc5c_args=deepcopy(args) 
        bc5c_args.guidee_data='BC5CDR-chem'
        modelDict['BC5CDR-chem']={'args':bc5c_args}
    if args.bc5_disease:
        bc5d_args=deepcopy(args)
        bc5d_args.guidee_data='BC5CDR-disease'
        modelDict['BC5CDR-disease']={'args':bc5d_args}
    if args.bc5:
        bc5_args=deepcopy(args) 
        bc5_args.guidee_data='BC5CDR'
        modelDict['BC5CDR']={'args':bc5_args}
    
    modelStart = time.time()
    modelClass = Model(args, wordEmbedding, seedV)
    
    for dataSet in modelDict:
        modelDict[dataSet]['summery']=dict()
        modelDict[dataSet]['CLWE']=modelClass.clwe(args=modelDict[dataSet]['args'],ID2char=ID2char)
        modelDict[dataSet]['WE']=modelClass.we(args=modelDict[dataSet]['args'])
        modelDict[dataSet]['model']=modelClass.model(args=modelDict[dataSet]['args'], 
                                                     X_embedded_data=modelDict[dataSet]['WE'], 
                                                     X_embedded_char=modelDict[dataSet]['CLWE'],
                                                     guideeInfo=None, 
                                                     summery=modelDict[dataSet]['summery'],
                                                     scopename=dataSet) #guideeInfo=None cuz in function we define
    dataNames = list()
    predResultSaver = dict()
    for dataSet in modelDict:
        modelDict[dataSet]['lossList'] = list()
        modelDict[dataSet]['f1ValList'] = list()
        modelDict[dataSet]['f1ValWOCRFList'] = list()
        modelDict[dataSet]['maxF1'] = 0.0
        modelDict[dataSet]['t_maxF1'] = 0.0
        modelDict[dataSet]['maxF1idx'] = 0
        modelDict[dataSet]['prevF1'] = 0.0
        modelDict[dataSet]['stop_counter'] = 0
        modelDict[dataSet]['early_stop'] = False
        modelDict[dataSet]['m_name'] = modelDict[dataSet]['args'].guidee_data

        dataNames.append(dataSet)
        
        modelDict[dataSet]['runner']=RunModel(model=modelDict[dataSet]['model'], args=modelDict[dataSet]['args'], 
                        ID2wordVecIdx=ID2wordVecIdx, ID2char=ID2char, 
                        expName=expName, m_name=modelDict[dataSet]['m_name'], m_train=m_train, m_dev=m_dev, m_test=m_test)

    # Get test result for each model
    for dataSet in modelDict:
        m_name = modelDict[dataSet]['args'].guidee_data
        print('===='+m_name.upper()+"_MODEL Test=====")
        with tf.Session(config=gpu_config) as sess:
            sess.run(tf.global_variables_initializer())
            loader = tf.train.Saver(max_to_keep=10000)
            loadpath = './modelSave/'+expName+'/'+m_name+'/bestModel/'
            
            if args.pretrained != 0:
                intOuts = dict()
                intOuts[m_test] = pickle.load(open(loadpath+'bestInouts.pickle','rb'))
            else:
                intOuts = None

            trsPara = pickle.load(open(loadpath+'trs_param.pickle','rb'))
            maxF1idx = pickle.load(open(loadpath+'maxF1idx.pickle','rb'))
            loader.restore(sess, tf.train.latest_checkpoint(loadpath)) 
            
            if modelDict[dataSet]['args'].tensorboard:
                tbWriter = tf.summary.FileWriter('test')
            else: tbWriter = None
            
            (t_predictionResult, t_prfValResult, t_prfValWOCRFResult,
             test_x, test_ans, test_len) = modelDict[dataSet]['runner'].dev1epoch(m_test, trsPara, sess, infoInput=intOuts, epoch=None)
