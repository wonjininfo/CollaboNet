import tensorflow as tf
import numpy as np
import random

from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell_impl

from ops.ops import *

def highway(input_, dim, num_layer=2):
    size = dim
    for i in range(num_layer):
        with tf.variable_scope("highway-%d" % i):
            W_p = tf.get_variable("W_p", [size, size])
            b_p = tf.get_variable("B_p", [1, size], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(input_, W_p) + b_p, "relu-proj")

            W_t = tf.get_variable("W_t", [size, size])
            b_t = tf.get_variable("B_t", [1, size], initializer=tf.constant_initializer(-2.0))
            transform = tf.nn.sigmoid(tf.matmul(input_, W_t) + b_t, "sigmoid-transform")
        
        input_ = tf.multiply(transform, proj) + tf.multiply(input_, 1 - transform)
    return input_, size

def mlp(input_, dim):
    n_hidden1 = int(dim*0.8)
    n_hidden2 = int(n_hidden1*0.8)
    n_out = int(n_hidden2*0.8)

    with tf.variable_scope("mlp"):
        h1 = tf.Variable(tf.random_normal([dim,n_hidden1]))
        h2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
        hout = tf.Variable(tf.random_normal([n_hidden2, n_out]))
    
        b1 = tf.Variable(tf.random_normal([n_hidden1]))
        b2 = tf.Variable(tf.random_normal([n_hidden2]))
        bout = tf.Variable(tf.random_normal([n_out]))

        layer1 = tf.add(tf.matmul(input_, h1), b1)
        layer1 = tf.nn.relu(layer1)
        layer2 = tf.add(tf.matmul(layer1, h2), b2)
        layer2 = tf.nn.relu(layer2)
        out_layer = tf.matmul(layer2, hout) + bout
    
    return out_layer, n_out 

def tfFC(input_, dim):
    n_hidden1 = int(dim*0.8)
    n_hidden2 = int(n_hidden1*0.8)
   
    #default active_fn = relu 
    h1 = tf.contrib.layers.fully_connected( 
            inputs=input_, num_outputs=n_hidden1)
    h2 = tf.contrib.layers.fully_connected( 
            inputs=h1, num_outputs=n_hidden2)

    return h2, n_hidden2

def normal_concat(input_, dim):
    return input_, dim
        
def concat_fc(input_, dim, method, layers=2):
    if method == 'highway':
        return highway(input_, dim, layers)
    elif method == 'mlp':
        return mlp(input_, dim)
    elif method == 'tfFC':
        return tfFC(input_, dim)
    elif method == 'normal':
        return normal_concat(input_, dim)
    else:
        print("concat_fc error")
