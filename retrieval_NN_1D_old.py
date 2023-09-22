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

time_stamps = [
    '2018-09-03_00_00_00',
    '2018-09-03_01_00_00',
    '2018-09-03_02_00_00',
    '2018-09-03_03_00_00',
    '2018-09-03_04_00_00',
    '2018-09-03_05_00_00',
    '2018-09-03_06_00_00',
    '2018-09-03_07_00_00',
    '2018-09-03_08_00_00',
    '2018-09-03_09_00_00',
    '2018-09-03_10_00_00',
    '2018-09-03_11_00_00',
    '2018-09-03_12_00_00',
    '2018-09-03_13_00_00',
    '2018-09-03_14_00_00',
    '2018-09-03_15_00_00',
    '2018-09-03_16_00_00',
    '2018-09-03_17_00_00',
    '2018-09-03_18_00_00',
    '2018-09-03_19_00_00',
    '2018-09-03_20_00_00',
    '2018-09-03_21_00_00',
    '2018-09-03_22_00_00',
    '2018-09-03_23_00_00',
    '2018-09-04_00_00_00',
    '2018-09-04_01_00_00',
    '2018-09-04_02_00_00',
    '2018-09-04_03_00_00',
    '2018-09-04_04_00_00',
    '2018-09-04_05_00_00',
    '2018-09-04_06_00_00',
    '2018-09-04_07_00_00',
    '2018-09-04_08_00_00',
    '2018-09-04_09_00_00',
    '2018-09-04_10_00_00',
    '2018-09-04_11_00_00',
    '2018-09-04_12_00_00',
    '2018-09-04_13_00_00',
    '2018-09-04_14_00_00',
    '2018-09-04_15_00_00',
    '2018-09-04_16_00_00',
    '2018-09-04_17_00_00',
    '2018-09-04_18_00_00',
    '2018-09-04_19_00_00',
    '2018-09-04_20_00_00',
    '2018-09-04_21_00_00',
    '2018-09-04_22_00_00',
    '2018-09-04_23_00_00',
    '2018-09-05_00_00_00',
    '2018-09-05_01_00_00',
    '2018-09-05_02_00_00',
    '2018-09-05_03_00_00']

path_to_V = '/home/seddik/Documents/workdir/WRF_Jebi/NN/'
path_to_V_non_scaled = '/home/seddik/Documents/workdir/WRF_Jebi/VISUAL/'
path_to_eta = '/home/seddik/Documents/workdir/WRF_Jebi/NN/' + time_stamps[0] + '/ZNU'

# The first element of V should always be the target variable
target = 'U'
key1 = 'U'
key2 = 'P_HYD'
key3 = 'QVAPOR'
V = [key1, key2, key3]
V_dict = {key1: 1, key2: 1, key3: 1}
# Do we use the distance to the neighboring points
include_distance = True

EPOCHS = 1000

required_neighbors = 8
num_training_samples = 500
num_test_samples = 1
num_vert_profiles = 5
batch_size = 50

num_training_time_stamps = len(time_stamps)
num_test_time_stamps = len(time_stamps)
num_vert_layers = 50

num_hidden_layers = 2
num_nodes = 32

# The hidden layers after concatenation
unified_num_hidden_layers = 1
unified_num_nodes = 68

eta = []
for i in range(num_vert_layers):
    file = str(i) + '.csv'
    data = load(path_to_eta, file, True)
    z = data.ZNU
    eta.append(z[0])

print(eta)


non_scaled_layers_files = []
for l in range(num_vert_layers):
    file = str(l) + '_interpol.csv'
    non_scaled_layers_files.append(file)

# Compute the number of inputs to the network
num_columns = 0
for item in V_dict.values():
    if item == 1:
        num_columns = num_columns + 1

num_columns = (num_columns*required_neighbors)
if include_distance:
    num_columns = num_columns + required_neighbors
num_columns = num_columns + 2
print('Total number of inputs to the network:')
print(num_columns)

num_examples = num_training_time_stamps * num_training_samples * num_vert_layers
print('Total number of examples used for training:')
print(num_examples)


NN_inputs = [[0 for j in range(num_columns)] for i in range(num_examples)]

ii = 0
for t in range(num_training_time_stamps):

    Data_holder = pd.DataFrame()
    var_idx = 0
    loc = 0
    for item in V_dict.values():
        if item == 1:
            path = path_to_V + time_stamps[t] + '/' + V[var_idx]
            f = str(0) + '.csv'
            raw_data = load(path, f, True)
            raw_data = raw_data.dropna()
            VAR = raw_data.pop(V[var_idx])
            Data_holder.insert(loc=loc, column=V[var_idx], value=VAR)
            X = raw_data[data_x]
            Y = raw_data[data_y]
            loc = loc + 1
        var_idx = var_idx + 1

    coordinates = [[0 for j in range(2)] for i in range(len(X))]
    for i in range(len(X)):
        coordinates[i][0] = X[i]
        coordinates[i][1] = Y[i]

    NeighborsTree = spatial.cKDTree(coordinates, leafsize=100)

    sample_data = raw_data.sample(n=num_training_samples)
    sample_data = sample_data.reset_index(drop=True)

    X_sample = sample_data[data_x]
    Y_sample = sample_data[data_y]

    # Get the non scaled values at each layer
    Non_scaled_at_layer = pd.DataFrame()
    for l in range(num_vert_layers):
        path = path_to_V_non_scaled + time_stamps[t] + '/' + target
        raw_data_non_scaled = load(path, non_scaled_layers_files[l], False)
        buffer = raw_data_non_scaled.pop(target)
        label = 'layer_' + str(l)
        Non_scaled_at_layer.insert(loc=l, column=label, value=buffer)

    coord_sample = [[0 for j in range(2)] for i in range(num_training_samples)]
    for i in range(num_training_samples):
        coord_sample[i][0] = X_sample[i]
        coord_sample[i][1] = Y_sample[i]

    for item in coord_sample:
        dd, loc = NeighborsTree.query(item, k=required_neighbors + 1)
        print("loc:", item)

        for l in range(num_vert_layers):
            # The non scaled value is used for the label
            label = 'layer_' + str(l)
            U_non_scaled = Non_scaled_at_layer[label]

            NN_inputs[ii][0] = U_non_scaled[loc[0]]

            var_idx = 0
            jj = 1
            for vv in V_dict.values():
                if vv == 1:
                    VAR = Data_holder[V[var_idx]]
                    idx = 1
                    for j in range(0, required_neighbors):
                        NN_inputs[ii][jj] = VAR[loc[idx]]
                        jj = jj + 1
                        idx = idx + 1
                var_idx = var_idx + 1

            if include_distance:
                idx = 1
                for j in range(0, required_neighbors):
                    NN_inputs[ii][jj] = dd[idx]
                    jj = jj + 1
                    idx = idx + 1

            NN_inputs[ii][num_columns - 1] = eta[l]
            ii = ii + 1

NN_columns = []
NN_columns.append('V')
var_idx = 0
for item in V_dict.values():
    if item == 1:
        for i in range(required_neighbors):
            label = V[var_idx] + '_' + str(i)
            NN_columns.append(label)
    var_idx = var_idx + 1

if include_distance:
    for i in range(required_neighbors):
        label = 'D_' + str(i)
        NN_columns.append(label)

NN_columns.append('eta')

NN_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
print(NN_dataset.tail())

train_dataset = NN_dataset.sample(frac=0.8, random_state=0)
print(train_dataset.tail())

test_dataset = NN_dataset.drop(train_dataset.index)

print('Train labels:')
train_labels = train_dataset.pop('V')
print(train_labels.tail())
print('Test labels:')
test_labels = test_dataset.pop('V')
print(test_labels.tail())

# The final input layers ready to be passed to the NN

# We support up to four input channels in addition to the (optional) distance and eta channels


def final_nn_inputs(dataset, v1, v2, v3, v4, v5, v6, d, eta, need_test, test_set=None, v1_test=None, v2_test=None,
                    v3_test=None, v4_test=None, v5_test=None, v6_test=None, d_test=None, eta_test=None):

    for i in range(required_neighbors):
        if include_distance:
            label = 'D_' + str(i)
            buffer1 = dataset.pop(label)
            if need_test:
                buffer2 = test_set.pop(label)
            d.insert(loc=i, column=label, value=buffer1)
            if need_test:
                d_test.insert(loc=i, column=label, value=buffer2)

        var_idx = 0
        for item in V_dict.values():
            if item == 1:
                label = V[var_idx] + '_' + str(i)
                buffer1 = dataset.pop(label)
                if need_test:
                    buffer2 = test_set.pop(label)
                if (var_idx + 1) == 1:
                    v1.insert(loc=i, column=label, value=buffer1)
                    if need_test:
                        v1_test.insert(loc=i, column=label, value=buffer2)
                if (var_idx + 1) == 2:
                    v2.insert(loc=i, column=label, value=buffer1)
                    if need_test:
                        v2_test.insert(loc=i, column=label, value=buffer2)
                if (var_idx + 1) == 3:
                    v3.insert(loc=i, column=label, value=buffer1)
                    if need_test:
                        v3_test.insert(loc=i, column=label, value=buffer2)
                if (var_idx + 1) == 4:
                    v4.insert(loc=i, column=label, value=buffer1)
                    if need_test:
                        v4_test.insert(loc=i, column=label, value=buffer2)
                if (var_idx + 1) == 5:
                    v5.insert(loc=i, column=label, value=buffer1)
                    if need_test:
                        v5_test.insert(loc=i, column=label, value=buffer2)
                if (var_idx + 1) == 6:
                    v6.insert(loc=i, column=label, value=buffer1)
                    if need_test:
                        v6_test.insert(loc=i, column=label, value=buffer2)
            var_idx = var_idx + 1

    buffer = dataset.pop('eta')
    eta.insert(loc=0, column='eta', value=buffer)

    if need_test:
        buffer = test_set.pop('eta')
        eta_test.insert(loc=0, column='eta', value=buffer)

    if need_test:
        description = 'train and test sets:'
    else:
        description = 'gen set:'

    var_idx = 0
    for item in V_dict.values():
        if item == 1:
            message = 'V' + str(var_idx+1) + ' ' + description
            print(message)
            if (var_idx + 1) == 1:
                print(v1.tail())
                if need_test:
                    print(v1_test.tail())
            if (var_idx + 1) == 2:
                print(v2.tail())
                if need_test:
                    print(v2_test.tail())
            if (var_idx + 1) == 3:
                print(v3.tail())
                if need_test:
                    print(v3_test.tail())
            if (var_idx + 1) == 4:
                print(v4.tail())
                if need_test:
                    print(v4_test.tail())
            if (var_idx + 1) == 5:
                print(v5.tail())
                if need_test:
                    print(v5_test.tail())
            if (var_idx + 1) == 6:
                print(v6.tail())
                if need_test:
                    print(v6_test.tail())
        var_idx = var_idx + 1

    if include_distance:
        message = 'Neighbors distance ' + description
        print(message)
        print(d.tail())
        if need_test:
            print(d_test.tail())

    message = 'eta ' + description
    print(message)
    print(eta.tail())
    if need_test:
        print(eta_test.tail())


V1_Train_set = pd.DataFrame()
V1_Test_set = pd.DataFrame()

V2_Train_set = pd.DataFrame()
V2_Test_set = pd.DataFrame()

V3_Train_set = pd.DataFrame()
V3_Test_set = pd.DataFrame()

V4_Train_set = pd.DataFrame()
V4_Test_set = pd.DataFrame()

V5_Train_set = pd.DataFrame()
V5_Test_set = pd.DataFrame()

V6_Train_set = pd.DataFrame()
V6_Test_set = pd.DataFrame()

D_Train_set = pd.DataFrame()
D_Test_set = pd.DataFrame()

ETA_Train_set = pd.DataFrame()
ETA_Test_set = pd.DataFrame()

final_nn_inputs(train_dataset, V1_Train_set, V2_Train_set, V3_Train_set, V4_Train_set, V5_Train_set, V6_Train_set,
                D_Train_set, ETA_Train_set, True, test_dataset, V1_Test_set, V2_Test_set, V3_Test_set, V4_Test_set,
                V5_Test_set, V6_Test_set, D_Test_set, ETA_Test_set)


def add_hidden_layers(prev_layer, n_hidden_layers, n_nodes, activation):
    x = Dense(num_nodes, activation=activation)(prev_layer)
    for nl in range(n_hidden_layers-1):
        x = Dense(n_nodes, activation=activation)(x)
    return x


def build_model():
    vec_in = []
    vec_channel = []
    var_idx = 0
    for item in V_dict.values():
        if item == 1:
            if (var_idx + 1) == 1:
                v1_input = Input(shape=(required_neighbors,), dtype='float32', name='v1_input')
                v1_channel = add_hidden_layers(v1_input, num_hidden_layers, num_nodes, 'relu')
                vec_in.append(v1_input)
                vec_channel.append(v1_channel)
            if (var_idx + 1) == 2:
                v2_input = Input(shape=(required_neighbors,), dtype='float32', name='v2_input')
                v2_channel = add_hidden_layers(v2_input, num_hidden_layers, num_nodes, 'relu')
                vec_in.append(v2_input)
                vec_channel.append(v2_channel)
            if (var_idx + 1) == 3:
                v3_input = Input(shape=(required_neighbors,), dtype='float32', name='v3_input')
                v3_channel = add_hidden_layers(v3_input, num_hidden_layers, num_nodes, 'relu')
                vec_in.append(v3_input)
                vec_channel.append(v3_channel)
            if (var_idx + 1) == 4:
                v4_input = Input(shape=(required_neighbors,), dtype='float32', name='v4_input')
                v4_channel = add_hidden_layers(v4_input, num_hidden_layers, num_nodes, 'relu')
                vec_in.append(v4_input)
                vec_channel.append(v4_channel)
            if (var_idx + 1) == 5:
                v5_input = Input(shape=(required_neighbors,), dtype='float32', name='v5_input')
                v5_channel = add_hidden_layers(v5_input, num_hidden_layers, num_nodes, 'relu')
                vec_in.append(v5_input)
                vec_channel.append(v5_channel)
            if (var_idx + 1) == 6:
                v6_input = Input(shape=(required_neighbors,), dtype='float32', name='v6_input')
                v6_channel = add_hidden_layers(v6_input, num_hidden_layers, num_nodes, 'relu')
                vec_in.append(v6_input)
                vec_channel.append(v6_channel)
        var_idx = var_idx + 1

    if include_distance:
        dist_input = Input(shape=(required_neighbors,), dtype='float32', name='dist_input')
        dist_channel = add_hidden_layers(dist_input, num_hidden_layers, num_nodes, 'relu')
        vec_in.append(dist_input)
        vec_channel.append(dist_channel)

    eta_input = Input(shape=(1,), dtype='float32', name='eta_input')
    eta_channel = add_hidden_layers(eta_input, num_hidden_layers, 4, 'relu')
    vec_in.append(eta_input)
    vec_channel.append(eta_channel)

    concat_layer = keras.layers.concatenate(vec_channel)

    x = add_hidden_layers(concat_layer, unified_num_hidden_layers, unified_num_nodes, 'relu')

    # The output
    main_output = Dense(1, name='main_output')(x)

    model = Model(inputs=vec_in, outputs=main_output)

    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mae', 'mse'])

    return model


model = build_model()
print(model.summary())

# Try out a dummy model
vector = []
var_idx = 0
for item in V_dict.values():
    if item == 1:
        if (var_idx + 1) == 1:
            example_batch_v1 = V1_Train_set[:10]
            vector.append(example_batch_v1)
        if (var_idx + 1) == 2:
            example_batch_v2 = V2_Train_set[:10]
            vector.append(example_batch_v2)
        if (var_idx + 1) == 3:
            example_batch_v3 = V3_Train_set[:10]
            vector.append(example_batch_v3)
        if (var_idx + 1) == 4:
            example_batch_v4 = V4_Train_set[:10]
            vector.append(example_batch_v4)
        if (var_idx + 1) == 5:
            example_batch_v5 = V5_Train_set[:10]
            vector.append(example_batch_v5)
        if (var_idx + 1) == 6:
            example_batch_v6 = V6_Train_set[:10]
            vector.append(example_batch_v6)
    var_idx = var_idx + 1

if include_distance:
    example_batch_dist = D_Train_set[:10]
    vector.append(example_batch_dist)

example_batch_eta = ETA_Train_set[:10]
vector.append(example_batch_eta)

example_result = model.predict(vector)
print(example_result)


# Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end='')


vec_train = []
vec_test = []
var_idx = 0
for item in V_dict.values():
    if item == 1:
        if (var_idx + 1) == 1:
            vec_train.append(V1_Train_set)
            vec_test.append(V1_Test_set)
        if (var_idx + 1) == 2:
            vec_train.append(V2_Train_set)
            vec_test.append(V2_Test_set)
        if (var_idx + 1) == 3:
            vec_train.append(V3_Train_set)
            vec_test.append(V3_Test_set)
        if (var_idx + 1) == 4:
            vec_train.append(V4_Train_set)
            vec_test.append(V4_Test_set)
        if (var_idx + 1) == 5:
            vec_train.append(V5_Train_set)
            vec_test.append(V5_Test_set)
        if (var_idx + 1) == 6:
            vec_train.append(V6_Train_set)
            vec_test.append(V6_Test_set)
    var_idx = var_idx + 1

if include_distance:
    vec_train.append(D_Train_set)
    vec_test.append(D_Test_set)

vec_train.append(ETA_Train_set)
vec_test.append(ETA_Test_set)

history = model.fit(
            x=vec_train,
            y=train_labels,
            epochs=EPOCHS, batch_size=batch_size, validation_split=0.2,
            verbose=0, callbacks=[PrintDot()]
        )

print('\n')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# Evaluate using test set
loss, mae, mse = model.evaluate(
                    x=vec_test,
                    y=test_labels, verbose=0
                )
print("Testing set Mean Abs Error: {:5.2f} m/a", format(mae))


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [m/a]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.show(block=False)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$(m/a)^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Vall Error')
    plt.legend()
    plt.ylim([0, 100])
    plt.show(block=False)


plot_history(history)

test_predictions = model.predict(vec_test).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [m/a]')
plt.ylabel('Predictions [m/a]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-50,plt.xlim()[1]])
plt.ylim([-50,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show(block=False)

error_test = test_predictions - test_labels
plt.figure()
plt.hist(error_test, bins=25)
plt.xlabel('Prediction Error [m/a]')
_ = plt.ylabel("Count")
plt.show(block=False)


# Compute the vertical distribution of V for random locations

if num_test_samples > 1:
    num_vert_profiles = 1

for p in range(num_vert_profiles):

    t = random.randint(0, num_test_time_stamps-1)
    print('----- Data sampled using time stamp:')
    print(time_stamps[t])

    num_examples = num_test_samples * num_vert_layers
    NN_inputs = [[0 for j in range(num_columns)] for i in range(num_examples)]

    Data_holder = pd.DataFrame()
    var_idx = 0
    loc = 0
    for item in V_dict.values():
        if item == 1:
            path = path_to_V + time_stamps[t] + '/' + V[var_idx]
            f = str(0) + '.csv'
            raw_data = load(path, f, True)
            raw_data = raw_data.dropna()
            VAR = raw_data.pop(V[var_idx])
            Data_holder.insert(loc=loc, column=V[var_idx], value=VAR)
            X = raw_data[data_x]
            Y = raw_data[data_y]
            loc = loc + 1
        var_idx = var_idx + 1

    coordinates = [[0 for j in range(2)] for i in range(len(X))]
    for i in range(len(X)):
        coordinates[i][0] = X[i]
        coordinates[i][1] = Y[i]

    NeighborsTree = spatial.cKDTree(coordinates, leafsize=100)

    sample_data = raw_data.sample(n=num_test_samples)
    sample_data = sample_data.reset_index(drop=True)
    print(sample_data.tail())

    X_sample = sample_data[data_x]
    Y_sample = sample_data[data_y]

    # Get the non scaled values at each layer
    Non_scaled_at_layer = pd.DataFrame()
    for l in range(num_vert_layers):
        path = path_to_V_non_scaled + time_stamps[t] + '/' + target
        raw_data_non_scaled = load(path, non_scaled_layers_files[l], False)
        buffer = raw_data_non_scaled.pop(target)
        label = 'layer_' + str(l)
        Non_scaled_at_layer.insert(loc=l, column=label, value=buffer)

    coord_sample = [[0 for j in range(2)] for i in range(num_test_samples)]
    for i in range(num_test_samples):
        coord_sample[i][0] = X_sample[i]
        coord_sample[i][1] = Y_sample[i]

    ii = 0
    for item in coord_sample:
        dd, loc = NeighborsTree.query(item, k=required_neighbors+1)
        print("loc:", item)

        for l in range(num_vert_layers):

            # The non scaled value is used for the label
            label = 'layer_' + str(l)
            U_non_scaled = Non_scaled_at_layer[label]

            NN_inputs[ii][0] = U_non_scaled[loc[0]]

            var_idx = 0
            jj = 1
            for vv in V_dict.values():
                if vv == 1:
                    VAR = Data_holder[V[var_idx]]
                    idx = 1
                    for j in range(0, required_neighbors):
                        NN_inputs[ii][jj] = VAR[loc[idx]]
                        jj = jj + 1
                        idx = idx + 1
                var_idx = var_idx + 1

            if include_distance:
                idx = 1
                for j in range(0, required_neighbors):
                    NN_inputs[ii][jj] = dd[idx]
                    jj = jj + 1
                    idx = idx + 1

            NN_inputs[ii][num_columns-1] = eta[l]
            ii = ii + 1

    NN_gen_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
    print(NN_gen_dataset.tail())

    V1_gen_set = pd.DataFrame()
    V2_gen_set = pd.DataFrame()
    V3_gen_set = pd.DataFrame()
    V4_gen_set = pd.DataFrame()
    V5_gen_set = pd.DataFrame()
    V6_gen_set = pd.DataFrame()
    D_gen_set = pd.DataFrame()
    ETA_gen_set = pd.DataFrame()

    final_nn_inputs(NN_gen_dataset, V1_gen_set, V2_gen_set, V3_gen_set, V4_gen_set, V5_gen_set, V6_gen_set, D_gen_set,
                    ETA_gen_set, False)

    gen_labels = NN_gen_dataset.pop('V')
    print('gen labels:')
    print(gen_labels.tail())

    vec_gen = []
    var_idx = 0
    for item in V_dict.values():
        if item == 1:
            if (var_idx + 1) == 1:
                vec_gen.append(V1_gen_set)
            if (var_idx + 1) == 2:
                vec_gen.append(V2_gen_set)
            if (var_idx + 1) == 3:
                vec_gen.append(V3_gen_set)
            if (var_idx + 1) == 4:
                vec_gen.append(V4_gen_set)
            if (var_idx + 1) == 5:
                vec_gen.append(V5_gen_set)
            if (var_idx + 1) == 6:
                vec_gen.append(V6_gen_set)
        var_idx = var_idx + 1

    if include_distance:
        vec_gen.append(D_gen_set)

    vec_gen.append(ETA_gen_set)

    compute_gen = model.predict(vec_gen).flatten()

    if num_test_samples > 1:
        plt.figure()
        plt.scatter(gen_labels, compute_gen)
        plt.xlabel('True Values [m/a]')
        plt.ylabel('Predictions [m/a]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-50, plt.xlim()[1]])
        plt.ylim([-50, plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show(block=False)

        error = compute_gen - gen_labels
        plt.figure()
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error [m/a]')
        _ = plt.ylabel("Count")
        plt.show()
    else:
        plt.figure()
        plt.gca().invert_yaxis()
        plt.plot(gen_labels, eta, 'b', label='WRF')
        plt.plot(compute_gen, eta, 'r', label='NN')
        plt.xlabel('V (m/a)')
        plt.ylabel(r'$\eta$')
        plt.legend(loc='lower left')
        if p == (num_vert_profiles - 1):
            plt.show()
        else:
            plt.show(block=False)
