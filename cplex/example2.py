# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:58:19 2017

@author: Biagio Brattoli

Optimization: Given two sets of points, find BETA points in the first set and BETA from the second set which
maximize the average distance between selected points from different classes and maximize the compactness
of the selected points inside each class

minimize sum_i sum_j y(i) y(j) I(i) I(j) D(i,j)
with:
    I is binary vector for selection
    y is {-1,1}
    i,j points in the first set
    score() distance to the nearest neightbors of the second set
"""

from pycpx import CPlexModel
import numpy as np
from time import time

def select_sequences_optimization(dist_set1,dist_set2,distance12,beta,normilize=[1,1]):
    S = dist_set1.shape[0]
    
    model = CPlexModel(verbosity=3)
    
    X1 = model.new((1, S), vtype=bool, name='x1')
    X2 = model.new((1, S), vtype=bool, name='x2')
    model.constrain(X1.sum() == beta)
    model.constrain(X2.sum() == beta)
    
    mask1 = X1.dot(X1.T)
    obj_1_0 = mask1.mult(dist_set1).sum()
    
    mask2 = X2.dot(X2.T)
    obj_1_1 = mask2.mult(dist_set2).sum()
    
    obj_1 = (obj_1_0 + obj_1_1)/2
    
    mask3 = X2.dot(X1.T)
    obj_2 = mask3.mult(distance12).sum()
    
    mod=model.minimize(obj_1/normilize[0]-obj_2/normilize[1])
    
    print('Error %.3f'%mod)
    
    X1 = np.asarray(model[X1])
    X2 = np.asarray(model[X2])
    return X1,X2


def test_optimization():
    from scipy.spatial.distance import cdist,squareform,pdist
    import matplotlib.pyplot as plt
    col = ['b','r']
    
    beta = 2
    normilize = [1,5]
    
    ###############  DATA  ##################
    points_A = np.random.normal(loc=[1,1],scale=0.5,size=(50,2))
    points_B = np.random.normal(loc=[3,3],scale=0.5,size=(50,2))
    
    dist_set1 = squareform(pdist(points_A,'euclidean'))
    dist_set2 = squareform(pdist(points_B,'euclidean'))
    distance12 = cdist(points_A,points_B)
    
    ###############  OPTIMIZATION  ##################
    t1 = time()
    X1, X2 = select_sequences_optimization(dist_set1,dist_set2,distance12,beta,normilize)
    print('Optimization done in %.2f'%(time()-t1))
    
    ###############  VISUALIZATION  ##################
    plt.subplot(1,2,1)
    plt.plot(points_A[:,0],points_A[:,1],col[0]+'.',markersize=15)
    plt.hold('On')
    plt.plot(points_B[:,0],points_B[:,1],col[1]+'.',markersize=15)
    plt.axis('Off')
    plt.title('All points')
    
    plt.subplot(1,2,2)
    plt.plot(points_A[X1==1,0],points_A[X1==1,1],col[0]+'.',markersize=15)
    plt.hold('On')
    plt.plot(points_B[X2==1,0],points_B[X2==1,1],col[1]+'.',markersize=15)
    plt.axis('Off')
    plt.title('Selected points')
    
    plt.show()
    

if __name__ == "__main__":
    test_optimization()
