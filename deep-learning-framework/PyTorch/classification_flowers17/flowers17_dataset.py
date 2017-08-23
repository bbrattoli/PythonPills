# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""

import os, sys, numpy as np
from scipy.io import loadmat
import torch.utils.data as data
from scipy.misc import imread, imresize

#os.chdir("/export/home/bbrattol/git/PythonPills/deep-learning-framework/PyTorch/")

class Flowers17(data.Dataset):
    def __init__(self,is_train=True,data_path='./data/',size=28):
        self.set = 'test'
        if is_train:
            self.set = 'train'
        self.info, self.images_dir = self.__dataset_info(data_path=data_path)
        self.N = len(self.info['x_'+self.set])
        self.size = size
    
    def __getitem__(self, index):
        image = self.info['x_'+self.set][index]
        label = self.info['y_'+self.set][index]
        img = imresize(imread(self.images_dir+image),[self.size,self.size])
        return img.astype(np.float32).transpose((2,0,1)), int(label-1)
        
    def __len__(self):
        return self.N
    
    def __dataset_info(self,data_path='./data/'):
        if not os.path.exists(data_path) or len(os.listdir(data_path))<2:
            os.system("wget http://www.robots.ox.ac.uk/~vgg/data/bicos/data/oxfordflower17.tar -P "+data_path)
            os.system("tar xf "+data_path+"oxfordflower17.tar -C "+data_path)
        
        labels = loadmat(data_path+'oxfordflower17/imagelabels.mat')['labels'].flatten()
        
        images_path = data_path+'oxfordflower17/jpg/'
        images = os.listdir(images_path)
        images = [f for f in images if '.jpg' in f]
        images.sort()
        images = np.array(images)
        
        N = len(images)
        sp = int(float(N)*0.8)
        sel = np.unique(np.linspace(0,N-1,sp)).astype(int)
        sel_test = np.arange(N)
        sel_test = np.delete(sel_test,sel)
        
        ret = {}
        ret['x_train'] = images[sel]
        ret['y_train'] = labels[sel]
        ret['x_test']  = images[sel_test]
        ret['y_test']  = labels[sel_test]
        
        return ret, images_path


