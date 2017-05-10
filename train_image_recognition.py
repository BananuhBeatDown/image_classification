# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:59:36 2017

@author: Matt Green
"""
import os
path = "C:/Users/Matt Green/Desktop/version-control/image_recognition"
os.chdir(path)

import pickle
import problem_unittests as tests
import helper
import tensorflow as tf

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# %%

# Build each neural network layer into a function.
# INPUT LAYER:

def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]), name='x')


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob') 


tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

# %%

# CONVOLUTION & MAX POOLING LAYER:
    
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    weight = tf.Variable(
                 tf.truncated_normal(
                     shape=[conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs],
                     stddev=0.1))
    bias = tf.Variable(tf.zeros(shape=conv_num_outputs))
    
    conv = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    hidden = tf.nn.relu(conv + bias)
    sub = tf.nn.max_pool(hidden,
                         ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                         strides=[1, pool_strides[0], pool_strides[1], 1],
                         padding='SAME')
    return sub


tests.test_con_pool(conv2d_maxpool)

# %%

# FLATTEN LAYER:

def flatten(x_tensor):
    shaped = x_tensor.get_shape().as_list()
    reshaped = tf.reshape(x_tensor, [-1, shaped[1] * shaped[2] * shaped[3]])
    return reshaped


tests.test_flatten(flatten)

# %%

# FULLY CONNECTED LAYER:

def fully_conn(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal(shape=[x_tensor.get_shape().as_list()[1], num_outputs], stddev=0.1)) 
    bias = tf.Variable(tf.zeros(shape=num_outputs))
    return tf.matmul(x_tensor, weight) + bias

tests.test_fully_conn(fully_conn)
