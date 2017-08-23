# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:01:24 2017

@author: bbrattol
"""

class MyModel(nn.Module):
    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Linear(pretrained_model.classifier[6].in_features,17)

    def forward(self, x):
        return self.last_layer(self.pretrained_model(x))

pretrained_model = torchvision.models.alexnet(pretrained=True)
model = MyModel(pretrained_model)

