# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:52:41 2017

@author: Biagio Brattoli
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

import tensorflow as tf

def rand_var(shape):
    return tf.truncated_normal(shape, stddev=0.01)

def const_var(shape):
    return tf.constant(0.1, shape=shape)

col = ['b','r']

# Data
points_A = np.random.normal(loc=[1,1],scale=0.5,size=(250,2))
points_B = np.random.normal(loc=[2,2],scale=0.5,size=(250,2))

train_A = points_A[0:200,:]
test_A  = points_A[200:,:]
train_B = points_B[0:200,:]
test_B  = points_B[200:,:]

train_X = np.concatenate([train_A,train_B],axis=0)
train_y = np.concatenate([np.ones(train_A.shape[0]),np.zeros(train_B.shape[0])],axis=0)

test_X = np.concatenate([test_A,test_B],axis=0)
test_y = np.concatenate([np.ones(test_A.shape[0]),np.zeros(test_B.shape[0])],axis=0)

# Network definition
with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, (None,2), name='x')
    y = tf.placeholder(tf.int64, (None,), name='y')

with tf.variable_scope('layer1'):
    h1_w = tf.Variable(rand_var((int(x.get_shape()[1]),64)), name='weight', trainable=True)
    h1_b = tf.Variable(const_var((64,)), name='bias', trainable=True)
    h1 = tf.add(tf.matmul(x,h1_w),h1_b, name='h1')

with tf.variable_scope('layer2'):
    h2_w = tf.Variable(rand_var((int(h1.get_shape()[1]),32)), name='weight', trainable=True)
    h2_b = tf.Variable(const_var((32,)), name='bias', trainable=True)
    h2 = tf.add(tf.matmul(h1,h2_w),h2_b, name='h2')

with tf.variable_scope('output'):
    out_w = tf.Variable(rand_var((int(h2.get_shape()[1]),2)), name='weight', trainable=True)
    out_b = tf.Variable(const_var((2,)), name='bias', trainable=True)
    out = tf.add(tf.matmul(h2,out_w),out_b, name='out')
    prob = tf.nn.softmax(out, name='prob')

graph = tf.get_default_graph()
config = tf.ConfigProto(log_device_placement=False,
                        allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3

sess = tf.Session(config=config)

# Loss definition
pred = tf.argmax(prob, dimension=1)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=y)
cost = tf.reduce_mean(cross_entropy)
train_op = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(cost)

correct_prediction = tf.equal(pred,tf.cast(y,tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())

# Train
N = train_X.shape[0]
batch_size = 100
B = N/batch_size

p = np.random.permutation(N)
train_X = train_X[p,:]
train_y = train_y[p]

for ii in xrange(5000):
    t1 = time()
    batch_idx = ii%B
    feed_dict_train = {x: train_X[batch_size*batch_idx:batch_size*(batch_idx+1)],
                       y: train_y[batch_size*batch_idx:batch_size*(batch_idx+1)]}
    
    _,acc,loss = sess.run([train_op,accuracy,cost], feed_dict=feed_dict_train)
    if ii%50==0:
        print 'Iter %d, Loss=%.3f, Train Accuracy= %.3f done in %.3f'%(ii,loss,acc,time()-t1)

# Train
feed_dict_train = {x: test_X, y: test_y}
_,acc,loss = sess.run([train_op,accuracy,cost], feed_dict=feed_dict_train)

print 'Test Loss=%.3f, Test Accuracy= %.3f'%(loss,acc)

