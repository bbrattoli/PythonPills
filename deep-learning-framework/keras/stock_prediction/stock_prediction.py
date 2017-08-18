# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:18:24 2017

@author: biagio
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def get_data(filename):
    dates = []
    prices = []
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    
    return np.asarray(prices)

def network(window,hidden=32):
    model = Sequential()
    model.add(LSTM(hidden, input_shape=(window,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def make_dataset(data,window):
    N = data.shape[0]
    X = []
    y = []
    for n in range(N-window-1):
        X.append(data[n:n+window])
        y.append(data[n+window])
    
    return np.asarray(X), np.asarray(y)

# Parameters
window = 10
data_file = 'aapl.csv'

# Initialization
data = get_data(data_file)
model = network(window)

X, y = make_dataset(data,window)
model.fit(X[:,:,np.newaxis], y, epochs=1000, batch_size=1, verbose=2)


pred = model.predict(X[:,:,np.newaxis])

plt.plot(np.arange(data.shape[0]),data,'b')
plt.hold('On')
plt.plot(np.arange(pred.shape[0])+10,pred,'-r')
plt.show()

