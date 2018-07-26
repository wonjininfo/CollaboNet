import tensorflow as tf
import numpy as np
from copy import deepcopy

from tensorflow.contrib import rnn
from tensorflow.contrib.tensorboard.plugins import projector

from ops.ops import *
from ops.embeddingOps import *
from ops.inputData import *
from model.sublayerFC import *

class Model:
    def __init__(self, args, wordEmbedding, seed):
        self.wordEmbedding = wordEmbedding
        self.X = tf.placeholder(tf.int32, [None, None], 'X')
        self.X_len = tf.placeholder(tf.int32, [None], 'X_len')
        self.X_char = tf.placeholder(tf.int32, [None, None, None], 'X_char')
        self.X_char_len = tf.placeholder(tf.int32, [None, None], 'X_char_len')
        self.maxLen = tf.placeholder(tf.int32, name='maxLen')
        self.lr = tf.placeholder(tf.float32, name='lr')        
        self.Y = tf.placeholder(tf.int32, [None, None], 'Y')
        self.infos  = tf.placeholder(tf.float32, [None, None, args.hidden_size*2], 'infos') # not use
        self.infos1 = tf.placeholder(tf.float32, [None, None, args.hidden_size*2], 'infos1')
        self.infos2 = tf.placeholder(tf.float32, [None, None, args.hidden_size*2], 'infos2')
        self.infos3 = tf.placeholder(tf.float32, [None, None, args.hidden_size*2], 'infos3')
        self.infos4 = tf.placeholder(tf.float32, [None, None, args.hidden_size*2], 'infos4')
        self.infos5 = tf.placeholder(tf.float32, [None, None, args.hidden_size*2], 'infos5')
        self.emb_dropout = tf.placeholder(tf.float32, name='emb_dropout')
        self.lstm_dropout = tf.placeholder(tf.float32, name='lstm_dropout')
        self.seed=seed

    def setArgs(self, args):
        self.args=args
        raise NotImplementedError

    def clwe(self,args,ID2char):
        X_char_embedded_data_temp = char_embedding(
                    inputs=self.X_char,
                    voca_size=len(ID2char),
                    embedding_dim=args.ce_dim,
                    initializer=None,
                    length=self.X_char_len,
                    charMaxLen=args.char_maxlen,
                    reuse=False,
                    trainable=True,
                    scope=args.guidee_data+'CE')
        X_char_embedded_data = X_char_embedded_data_temp 
            
        if args.clwe_method == 'biLSTM':
            with tf.variable_scope(args.guidee_data+'clwe_bi-LSTM'):        
                temp_char_emb = tf.reshape(X_char_embedded_data, [-1, args.char_maxlen, args.ce_dim], name='temp_char_emb')
                temp_char_len = tf.reshape(self.X_char_len, [-1], name='temp_char_len')
                self.ce_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=args.clwe_dim, 
                                                                state_is_tuple=True)
                self.ce_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=args.clwe_dim, 
                                                                state_is_tuple=True)
                self.ce_outputs, self.ce_states = tf.nn.bidirectional_dynamic_rnn(
                                                        self.ce_cell_fw, self.ce_cell_bw,
                                                        temp_char_emb,
                                                        sequence_length=temp_char_len,
                                                        dtype=tf.float32)
                self.temp = tf.concat([self.ce_states[0][1], self.ce_states[1][1]], axis=1, name='x_embchar_concat')
            X_embedded_char = tf.reshape(self.temp, [-1,self.maxLen, args.clwe_dim*2], name='X_embchar')

        elif args.clwe_method == 'CNN':
            filter_size = [3,5,7] 
            temp = list()
            with tf.variable_scope(args.guidee_data+"CLWE_CNN"):
                for fs in filter_size:  
                    with tf.variable_scope("clwe_CNN-%s" % fs):
                        filter_shape = [1, fs, args.ce_dim, args.clwe_dim]
                        W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.3, seed=self.seed), name='W')
                        b = tf.get_variable(initializer=tf.constant(0.0, shape=[args.clwe_dim]), name='b')
                        conv = tf.nn.conv2d(
                                X_char_embedded_data,
                                W,
                                strides=[1,1,1,1],
                                padding="VALID",
                                data_format='NHWC',
                                name="conv-char")
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        pooled = tf.nn.max_pool(
                                    h,
                                    ksize=[1, 1, args.char_maxlen-fs+1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    data_format='NHWC',
                                    name="CLWEpool")
                        pool_flat = tf.reshape(pooled,[-1, self.maxLen, args.clwe_dim],name='cnn_rs')
                        pool_flat_d = tf.nn.dropout(x=pool_flat, keep_prob=1-self.emb_dropout, name='clwe_drop', seed=self.seed)
                        temp.append(pool_flat_d)
                X_embedded_char = tf.concat([temp[0], temp[1], temp[2]],axis=2,name='X_embchar')
        return X_embedded_char
        
    def we(self,args):
        args=deepcopy(args)
        X_embedded_data_temp, embedding_table = embedding_lookup(
                    inputs=self.X,
                    voca_size=self.wordEmbedding.shape[0],
                    initializer=self.wordEmbedding,
                    trainable=True,
                    scope=args.guidee_data+'WE',
                    reuse=False)
        X_embedded_data = X_embedded_data_temp 
        
        return X_embedded_data

    def model(self, args, X_embedded_data, X_embedded_char, guideeInfo, summery, scopename='main'):
        """
        guideeInfo : [mandatory] additional infos (logits) from other models.
        scopename : scope name for each model.
        """
        
        self.summery=summery

        with tf.variable_scope(scopename) as scope:
            self.infos1_w = tf.Variable(initial_value=1.0, name='infos1_w', trainable=True)
            self.infos2_w = tf.Variable(initial_value=1.0, name='infos2_w', trainable=True)
            self.infos3_w = tf.Variable(initial_value=1.0, name='infos3_w', trainable=True)
            self.infos4_w = tf.Variable(initial_value=1.0, name='infos4_w', trainable=True)
            self.infos5_w = tf.Variable(initial_value=1.0, name='infos5_w', trainable=True)
            
            inf1 = tf.multiply(self.infos1_w, self.infos1, name='mul1')
            inf2 = tf.multiply(self.infos2_w, self.infos2, name='mul2')
            inf3 = tf.multiply(self.infos3_w, self.infos3, name='mul3')
            inf4 = tf.multiply(self.infos4_w, self.infos4, name='mul4')
            inf5 = tf.multiply(self.infos5_w, self.infos5, name='mul5')
            
            inf1 = tf.expand_dims(inf1,0) 
            inf2 = tf.expand_dims(inf2,0) 
            inf3 = tf.expand_dims(inf3,0) 
            inf4 = tf.expand_dims(inf4,0) 
            inf5 = tf.expand_dims(inf5,0)
            
            infos_concat = tf.concat([inf1, inf2, inf3, inf4, inf5], axis=0)
            infos_concat_e = tf.expand_dims(infos_concat, 0)
            infos_pooled = tf.nn.max_pool3d(
                        infos_concat_e,
                        ksize=[1, 5, 1, 1, 1],
                        strides=[1, 1, 1, 1, 1],
                        padding='VALID',
                        name="infos_pool")
            infos_pooled = tf.reshape(infos_pooled,[-1, self.maxLen, args.hidden_size*2])
            guideeInfo=infos_pooled

            concat = tf.concat([X_embedded_data, X_embedded_char, guideeInfo], 
                                axis=2, name='inputEmbeddingsConcat')
            dim = concat.get_shape().as_list()[2]
            concat_for_fc = tf.reshape(concat, [-1,dim], name='concat_fc')
            concat_for_fc, n_out = concat_fc(concat_for_fc, dim, args.fc_method, args.mlp_layer)
            self.X_embedded_concat = tf.reshape(concat_for_fc, [-1, self.maxLen, n_out], name='X_ec')
            
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=args.hidden_size,
                                                       state_is_tuple=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=args.hidden_size, 
                                                            state_is_tuple=True)
            self.outputs, self.states = tf.nn.bidirectional_dynamic_rnn(
                                                    cell_fw, cell_bw,
                                                    self.X_embedded_concat,
                                                    sequence_length=self.X_len,
                                                    dtype=tf.float32)
            self.outputs_fw, self.outputs_bw = self.outputs
            
            outputs_fw_t = tf.nn.dropout(x=self.outputs_fw, keep_prob=1-self.lstm_dropout, name='output_drop_fw', seed=self.seed)
            outputs_bw_t = tf.nn.dropout(x=self.outputs_bw, keep_prob=1-self.lstm_dropout, name='output_drop_bw', seed=self.seed)
            self.outputs_concat=tf.concat([outputs_fw_t, outputs_bw_t], axis=2, name='out_c')
            
            self.fc_outputs = tf.contrib.layers.fully_connected(
                                inputs=self.outputs_concat, num_outputs=args.num_class,
                                activation_fn=None)
            self.logits = tf.reshape(self.fc_outputs, 
                                [-1, self.maxLen, args.num_class], name='logits')
            self.weights = tf.sequence_mask(lengths=self.X_len, dtype=tf.float32, name='weights')
            self.sequence_loss = tf.contrib.seq2seq.sequence_loss(
                                logits=self.logits, targets=self.Y, weights=self.weights, name='seq_loss')
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                                                        self.logits, self.Y, self.X_len)
            self.loss=tf.reduce_mean(self.sequence_loss-args.loss_weight*self.log_likelihood, name='loss')
            self.train = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.prediction = tf.argmax(self.logits, axis=2, name='prediction')

            if args.tensorboard:
                with tf.variable_scope('summeries'):
                    tf.summary.scalar('loss',self.loss) # so train only
                self.variable_summaries(self.outputs_concat_tmp)
                self.summaryMerged = tf.summary.merge_all()

        return self

    # for tensorboard
    def variable_summaries(self, var): #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        with tf.name_scope('VarSummaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
