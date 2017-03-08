# -*- coding: utf-8 -*-
"""
Created on Tue Mar 8 2017

@author: Biagio Brattoli
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sys

sys.path.append("/export/home/bbrattol/caffe_compvisgpu03/python")
import caffe

#import os
#os.chdir('/net/hciserver03/storage/bbrattol/git/PythonPills/caffe/')

# Data
points_A = np.random.normal(loc=[1,1],scale=0.5,size=(600,2))
points_B = np.random.normal(loc=[3,3.5],scale=1.0,size=(600,2))
points_C = np.random.normal(loc=[0.5,3.5],scale=0.8,size=(600,2))

train_A = points_A[0:500,:]
test_A  = points_A[500:,:]
train_B = points_B[0:500,:]
test_B  = points_B[500:,:]
train_C = points_C[0:500,:]
test_C  = points_C[500:,:]

train_X = np.concatenate([train_A,train_B,train_C],axis=0)
train_y = np.concatenate([0*np.ones(train_A.shape[0]),
                          1*np.ones(train_B.shape[0]),
                          2*np.ones(train_C.shape[0])],axis=0)

test_X = np.concatenate([test_A,test_B,test_C],axis=0)
test_y = np.concatenate([0*np.ones(test_A.shape[0]),
                         1*np.ones(test_B.shape[0]),
                         2*np.ones(test_C.shape[0])],axis=0)

# Network definition
#caffe.set_device(gpu_id)
caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
net = solver.net
testnet = solver.test_nets[0]

# Train
N = train_X.shape[0]
batch_size = 100
B = N/batch_size

p = np.random.permutation(N)
train_X = train_X[p,:]
train_y = train_y[p]

for ii in xrange(10000):
    t1 = time()
    batch_idx = ii%B
    x = train_X[batch_size*batch_idx:batch_size*(batch_idx+1)]
    y = train_y[batch_size*batch_idx:batch_size*(batch_idx+1)]
    net.blobs['data'].data[...]  = x
    net.blobs['label'].data[...] = y
    solver.step(1)

# Test
testnet.blobs['data'].reshape(test_X.shape[0],test_X.shape[1])
testnet.blobs['label'].reshape(test_X.shape[0])
testnet.blobs['data'].data[...]  = test_X
testnet.blobs['label'].data[...] = test_y
out = testnet.forward()
acc = out['accuracy']
loss = out['loss']
print 'Test Loss=%.3f, Test Accuracy= %.3f'%(loss,acc)


# Plotting decision regions
x_min, x_max = test_X[:, 0].min() - 1, test_X[:, 0].max() + 1
y_min, y_max = test_X[:, 1].min() - 1, test_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

#f, axarr = plt.subplots(1,1, sharex='col', sharey='row', figsize=(10, 8))
X = np.c_[xx.ravel(), yy.ravel()]
testnet.blobs['data'].reshape(X.shape[0],X.shape[1])
testnet.blobs['label'].reshape(X.shape[0])
testnet.blobs['data'].data[...]  = X
out = testnet.forward()
Z = out['probs']
Z = np.argmax(Z,axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y, alpha=0.8)
plt.set_title('Decision boundaries')

plt.show()

