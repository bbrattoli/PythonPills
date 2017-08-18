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

os.chdir("/export/home/bbrattol/git/PythonPills/deep-learning-framework/PyTorch/")

from flowers17_dataset import Flowers17
from Architecture import CNN

USE_GPU = True

train_data = Flowers17(True)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=4)


test_data = Flowers17(False)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=100, 
                                           shuffle=True,
                                           num_workers=4)



num_epochs = 10

cnn = CNN(17)
if USE_GPU:
    cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

dtype = torch.FloatTensor

#data_iter = iter(train_loader)
#images, labels = data_iter.next()
#images = Variable(images)#.cuda()
#labels = Variable(labels)#.cuda()

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        if USE_GPU:
            images = images.cuda()
            labels = labels.cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, loss.data[0]))

# Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
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

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')