from __future__ import absolute_import, division, print_function

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import spatial
import gc
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from datetime import datetime
from functools import partial


def load(path1, path2, print_path):
    csv_path = os.path.join(path1, path2)
    if (print_path):
        print(csv_path)
    return pd.read_csv(csv_path)


data_x = 'longitude'
data_y = 'latitude'

time_stamps = ['2018-09-03_00_00_00']

path_to_V = '/home/seddik/Documents/workdir/WRF_Jebi/NN/'
path_to_V_non_scaled = '/home/seddik/Documents/workdir/WRF_Jebi/VISUAL/'
path_to_eta = '/home/seddik/Documents/workdir/WRF_Jebi/NN/' + time_stamps[0] + '/ZNU'

target = 'U'
key1 = 'U'
key2 = 'P_HYD'
key3 = 'QVAPOR'
V = [key1, key2, key3]

required_neighbors = 4

path = path_to_V + time_stamps[0] + '/' + V[0]
f = str(0) + '.csv'
raw_data = load(path, f, True)
raw_data = raw_data.dropna()
print(raw_data.tail())


X = raw_data[data_x]
Y = raw_data[data_y]

coordinates = [[0 for j in range(2)] for i in range(len(X))]
for i in range(len(X)):
    coordinates[i][0] = X[i]
    coordinates[i][1] = Y[i]

NeighborsTree = spatial.cKDTree(coordinates, leafsize=100)

for item in coordinates:
    dd, loc = NeighborsTree.query(item, k=required_neighbors + 1)
    print("loc:", item)
    for j in range(0, required_neighbors+1):
        print(X[loc[j]])
    break