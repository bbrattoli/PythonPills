# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:58:19 2017

@author: Biagio Brattoli

Optimization: Given two sets of points, find BETA points in the first set which
maximize the average distance from the nearest neighbors in the second set and maximize the compactness
of the selected set of points

minimize sum_i sum_j I(i) I(j) D(i,j) - sum_i I(i) score(i), 
with:
    I is binary vector for selection
    i,j points in the first set
    score() distance to the nearest neightbors of the second set
"""

from pycpx import CPlexModel
import numpy as np
from time import time

def select_sequences_optimization(dist_intra,dist_inter,beta,normilize=[1,1], NN=10):
    if dist_intra.shape[0]!=dist_inter.shape[0]:
        print 'dist_intra [NP,NP], dist_inter [NP,NN]'
        return 
    
    NP = dist_intra.shape[0]
    
    dist_inter = np.sort(dist_inter,axis=1)
    D = dist_inter[:,0:5].mean(axis=1)
    
    model = CPlexModel(verbosity=3)
    X = model.new((NP), vtype=bool, name='x')
    model.constrain(X.sum() == beta)
    
    mask = X.T.dot(X)
    obj_aux1 = mask.mult(dist_intra)
    obj_1 = obj_aux1.sum()/(beta**2)
    
    obj_2 = X.mult(D).sum()/beta
    
    mod=model.minimize(obj_1/normilize[0]-obj_2/normilize[1])
    print('Error %.3f'%mod)
    
    X = np.asarray(model[X])
    return X


def test_optimization():
    from scipy.spatial.distance import cdist,squareform,pdist
    import matplotlib.pyplot as plt
    col = ['b','r']
    
    beta = 5
    normilize = [1,5]
    
    ###############  DATA  ##################
    points_A = np.random.normal(loc=[1,1],scale=0.5,size=(20,2))
    points_B = np.random.normal(loc=[2,2],scale=0.5,size=(50,2))
    
    dist_intra = squareform(pdist(points_A,'euclidean'))
    dist_inter = cdist(points_A,points_B)
    
    ###############  OPTIMIZATION  ##################
    t1 = time()
    X = select_sequences_optimization(dist_intra,dist_inter,beta,normilize)
    print('Optimization done in %.2f'%(time()-t1))
    print '%.2f'%(dist_intra[X==1,:][:,X==1].mean())
    
    ###############  VISUALIZATION  ##################
    plt.subplot(1,2,1)
    plt.plot(points_A[:,0],points_A[:,1],col[0]+'.',markersize=15)
    plt.hold('On')
    plt.plot(points_B[:,0],points_B[:,1],col[1]+'.',markersize=15)
    plt.axis('Off')
    plt.title('All points')
    
    plt.subplot(1,2,2)
    plt.plot(points_A[X==1,0],points_A[X==1,1],col[0]+'.',markersize=15)
    plt.hold('On')
    plt.plot(points_B[:,0],points_B[:,1],col[1]+'.',markersize=15)
    plt.axis('Off')
    plt.title('Selected points')
    
    plt.show()
    

if __name__ == "__main__":
    test_optimization()
