# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""

import os, sys, numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

os.chdir("/export/home/bbrattol/git/PythonPills/deep-learning-framework/PyTorch/classification_flowers17/custom_cnn")

sys.path.append('../')
from flowers17_dataset import Flowers17
from Architecture import CNN

USE_GPU = True
SIZE = 28
LR = 0.001
num_epochs = 20
classes = 17

############## DATA LOADER ###############
train_data = Flowers17(is_train=True,data_path='../data/',size=SIZE)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=4)


test_data = Flowers17(is_train=False,data_path='../data/',size=SIZE)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=4)

############## INITIALIZE CNN AND SOLVER ###############
cnn = CNN(classes)
if USE_GPU:
    cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

dtype = torch.FloatTensor

#data_iter = iter(train_loader)
#images, labels = data_iter.next()
#images = Variable(images)#.cuda()
#labels = Variable(labels)#.cuda()

############## TRAINING ###############
# Train the Model
for epoch in range(num_epochs):
    accuracy = []
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        if USE_GPU:
            images = images.cuda()
            labels = labels.cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        
        _, predicted = torch.max(outputs.data, 1)
        accuracy.append(accuracy_score(labels.cpu().data.numpy(),predicted.cpu().numpy()))
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print ('Epoch [%d/%d], Loss: %.4f, Accuracy %.1f%%' %(epoch+1, num_epochs, loss.data[0],100*np.mean(accuracy)))

############## TESTING ###############
# Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
accuracy = []
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    if USE_GPU:
        images = images.cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    accuracy.append(accuracy_score(labels.numpy(),predicted.cpu().numpy()))

print('Test Accuracy of the model on the 10000 test images: %.1f%% (%.1f%%)' % (100 * correct / total, 100*np.mean(accuracy)))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')