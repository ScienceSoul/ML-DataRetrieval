from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import pandas as pd
from scipy import spatial
import numpy as np
import random
import sys

# from mpl_toolkits.basemap import Basemap
# import gc

from math import *
from load_model import *

data_x = 'longitude'
data_y = 'latitude'

# The target value is the variable for which the network is trained
# to compute
target = 'V'
key1 = 'V'
key2 = 'PRESSURE'
key3 = 'GEOPOT'
key4 = 'COR_EAST'
key5 = 'DRY_MASS'
V = [key1, key2, key3, key4, key5]
V_dict = {key1: 1, key2: 1, key3: 1, key4: 1, key5: 0}

# Do we use the distance to the neighboring points
include_distance = True

required_neighbors = 8
compute_kdTree_from_global_grid = True

# The order of the learning
#   0 -> neighbors values will be used as inputs and the target is a value at a given layer
#   1 -> difference to neighbors will be used as inputs and for the targets at a given layer
order_of_learning = 1

# The number of grid points away of a sampled point until which grid points
# are acquired. Only for first order learning
number_grid_points_apart = 10

# Constrain the target to value higher than LOWER_LIMIT
constrained_target = False
LOWER_LIMIT = 1.5

############################################################################################
# A special mode when the target is not available in the input data but is computed on 
# the fly.
# A depedency dictionary must be provided in order to specify the input variables that are
# used to compute the target
# A constrain flag can also be provided in order to constrain an area where the target is 
# computed. In that case, the input variable to use as constrain must be provided.
# The way the target is computed is defined by its name.

compute_target = False
target_depedencies = ["U", "V"]
compute_target_constrain = False
compute_target_constrain_name = "REL_VERT_VORT"
compute_target_lower_limit = 1.5
############################################################################################

EPOCHS = 100

num_training_samples = 20
time_stamp_sampling_times = 1

batch_size = 50

# Optional timestamp to ignore during training
ignore_timestamp = False
ignored_timestamp = -1

activation_func = 'relu'
regularization_fac = 0.0

def load(path1, path2, print_path):
    csv_path = os.path.join(path1, path2)
    if print_path:
        print(csv_path)
    return pd.read_csv(csv_path)


def get_eta(num_vert_layers, path_to_eta):
    eta = []
    for i in range(num_vert_layers):
        file = str(i) + '.csv'
        data = load(path_to_eta, file, True)
        z = data.ZNU
        eta.append(z[0])
    return eta


def get_non_scaled_layers_files(num_vert_layers):
    non_scaled_layers_files = []
    for l in range(num_vert_layers):
        if target == 'U' or target == 'V':
            file = str(l) + '_interpol.csv'
        else:
            file = str(l) + '.csv'
        non_scaled_layers_files.append(file)
    return non_scaled_layers_files


def compute_num_columns():
    num_columns = 0
    num_active_vars = 0
    use_coriolis = 0
    use_dry_mass = 0
    var_idx = 0
    for item in V_dict.values():
        if order_of_learning == 1:
            if V[var_idx] == 'COR_NORTH':
                if item == 1:
                    use_coriolis = use_coriolis + 1
                var_idx = var_idx + 1
                continue
            if V[var_idx] == 'COR_EAST':
                if item == 1:
                    use_coriolis = use_coriolis + 1
                var_idx = var_idx + 1
                continue
            if V[var_idx] == 'DRY_MASS':
                if item == 1:
                    use_dry_mass = use_dry_mass + 1
                var_idx = var_idx + 1
                continue

        if item == 1:
            num_active_vars = num_active_vars + 1
        var_idx = var_idx + 1

    if order_of_learning == 0:
        num_columns = (num_active_vars * required_neighbors)
        if include_distance:
            num_columns = num_columns + required_neighbors

        # Add the target value + eta
        num_columns = num_columns + 2
    elif order_of_learning == 1:
        if first_order_learn_mode == 1 or first_order_learn_mode == 2:
            num_columns = num_active_vars
        elif first_order_learn_mode == 3:
            var_idx = 0
            v = 0
            for item in V_dict.values():
                if V[var_idx] == target:
                    num_active_vars = num_active_vars - 1
                    v = 2
                var_idx = var_idx + 1
            num_columns = v + num_active_vars

        if include_distance:
            num_columns = num_columns + 1

        # Add the target value + compass + eta + coriolis force +
        # dry mass pressure if required
        num_columns = num_columns + 3 + use_coriolis + use_dry_mass

    print('Total number of inputs to the network:')
    print(num_columns - 1)
    return num_columns


def define_nn_columns():
    nn_columns = []

    # Name the target column(s) with a hard coded value(s)
    nn_columns.append('TAR')

    var_idx = 0
    for item in V_dict.values():
        if item == 1:
            if order_of_learning == 0:
                for i in range(required_neighbors):
                    label = V[var_idx] + '_' + str(i)
                    nn_columns.append(label)
            elif order_of_learning == 1:
                if V[var_idx] == 'COR_NORTH' or V[var_idx] == 'COR_EAST':
                    label = V[var_idx] + '_' + str(0)
                    nn_columns.append(label)
                    var_idx = var_idx + 1
                    continue
                if V[var_idx] == 'DRY_MASS':
                    label = V[var_idx] + ' ' + str(0)
                    nn_columns.append(label)
                    var_idx = var_idx + 1
                    continue

                if first_order_learn_mode == 1 or first_order_learn_mode == 2:
                    label = V[var_idx] + '_' + str(0)
                    nn_columns.append(label)
                elif first_order_learn_mode == 3:
                    if V[var_idx] == target:
                        for i in range(2):
                            label = V[var_idx] + '_' + str(i)
                            nn_columns.append(label)
                    else:
                        label = V[var_idx] + '_' + str(0)
                        nn_columns.append(label)

        var_idx = var_idx + 1

    if include_distance:
        if order_of_learning == 0:
            for i in range(required_neighbors):
                label = 'D_' + str(i)
                nn_columns.append(label)
        elif order_of_learning == 1:
            label = 'D_' + str(0)
            nn_columns.append(label)

    if order_of_learning == 1:
        label = 'COMP_' + str(0)
        nn_columns.append(label)

    nn_columns.append('eta')
    return nn_columns


def compute_bearing(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    return atan2(y, x)


def compass(bearing):
    bearing_degress = bearing / pi*180
    if bearing_degress < 0:
        bearing_degress = bearing_degress + 360
    compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    compass_lookup = round(bearing_degress / 45)
    if compass_lookup == 8:
        compass_lookup = 0
    return compass_lookup


def loop_common_1(data_holder, t, path_to_V, time_stamps, first_time):

    var_idx = 0
    loc = 0
    for item in V_dict.values():
        if item == 1:
            f = str(0) + '.csv'
            path = path_to_V + time_stamps[t] + '/' + V[var_idx]
            raw_data = load(path, f, True)
            raw_data = raw_data.dropna()
            VAR = raw_data.pop(V[var_idx])
            data_holder.insert(loc=loc, column=V[var_idx], value=VAR)
            if compute_kdTree_from_global_grid:
                X = raw_data[data_x]
                Y = raw_data[data_y]
            loc = loc + 1
        var_idx = var_idx + 1
    return X, Y, raw_data


def loop_common_2(x, y, return_coord):
    coordinates = [[0 for j in range(2)] for i in range(len(x))]
    for i in range(len(x)):
        coordinates[i][0] = x[i]
        coordinates[i][1] = y[i]

    NeighborsTree = spatial.cKDTree(coordinates, leafsize=100)
    if return_coord:
        return NeighborsTree, coordinates
    else:
        return NeighborsTree


def loop_common_4(NN_inputs, eta, dd, ii, jj, num_columns, l, test,
                  neighbor_idx=None, compass=None, compute_2d_map=None, map_layer=None):
    # Add the distance to neighbor(s)
    if include_distance:
        if order_of_learning == 0:
            idx = 1
            for j in range(0, required_neighbors):
                NN_inputs[ii][jj] = dd[idx]
                jj = jj + 1
                idx = idx + 1
        elif order_of_learning == 1:
            NN_inputs[ii][jj] = dd[neighbor_idx]
            jj = jj + 1

    # Add the compass when first order learning
    if order_of_learning == 1:
        NN_inputs[ii][jj] = compass
        jj = jj + 1

    # Add eta
    if test:
        if compute_2d_map:
            NN_inputs[ii][num_columns - 1] = eta[map_layer]
        else:
            NN_inputs[ii][num_columns - 1] = eta[l]
    else:
        NN_inputs[ii][num_columns - 1] = eta[l]


# The final input layers ready to be passed to the NN

# We support up to four input channels in addition to the (optional) distance and eta channels

def final_nn_inputs(dataset, num_neighbors, v1, v2, v3, v4, v5, v6, d, eta, need_test, test_set=None,
                    v1_test=None, v2_test=None, v3_test=None, v4_test=None, v5_test=None,
                    v6_test=None, d_test=None, eta_test=None):

    for i in range(num_neighbors):
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


def average_over_outputs(predictions, labels):
    buffer = labels.to_numpy()
    mean_predict = predictions.mean(axis=1)
    mean_labels = buffer.mean(axis=1)
    return mean_predict, mean_labels


def get_sample_with_constrain(raw_data, non_scaled_at_layer, constrain, path_to_V_non_scaled=None, 
                              time_stamps=None, t=None):
    if compute_target:
        path = path_to_V_non_scaled + time_stamps[t] + '/' + compute_target_constrain_name
        file = str(0) + '.csv'
        raw = load(path, file, False)
        var = raw.pop(compute_target_constrain_name)
    else:
        var  = non_scaled_at_layer['layer_0']

    constrained_indexes = []
    constrained_buffer = pd.DataFrame()
    for i in range(len(var)):
        if var[i] >= constrain:
            constrained_indexes.append(i)

    if len(constrained_indexes) == 0:
        print("No value found that satisfy the constrain.")
        sys.exit()

    constrained_buffer.insert(loc=0, column='INDEX', value=constrained_indexes)
    print("Length of constrained target buffer: ", len(constrained_indexes))
    return constrained_buffer

def compute_turb_moment_fluxes(non_scaled_at_layer, num_vert_layers, path_to_V_non_scaled, time_stamps, t):
    if compute_target_constrain:
        path3 = path_to_V_non_scaled + time_stamps[t] + '/' + compute_target_constrain_name
        file = str(0) + '.csv'
        raw3 = load(path3, file, False)
        buff3 = raw3.pop(compute_target_constrain_name)
        constrained_indexes = []
        constrained_buffer = pd.DataFrame()
        for i in range(len(buff3)):
            if buff3[i] >= compute_target_lower_limit:
                constrained_indexes.append(i)

        if len(constrained_indexes) == 0:
            print("No value found that satisfy the constrain in computed target.")
            print("Got: ", len(constrained_indexes))
            sys.exit()

        constrained_buffer.insert(loc=0, column='INDEX', value=constrained_indexes)
        index = constrained_buffer['INDEX']

    for l in range(num_vert_layers):
        path1 = path_to_V_non_scaled + time_stamps[t] + '/' + target_depedencies[0]
        path2 = path_to_V_non_scaled + time_stamps[t] + '/' + target_depedencies[1]
        if target_depedencies[0] == 'U' or target_depedencies[0] == 'V':
            file1 = str(l) + '_interpol.csv'
        else:
            file1 = str(l) + '.csv'
        if target_depedencies[1] == 'U' or target_depedencies[1] == 'V':
            file2 = str(l) + '_interpol.csv'
        else:
            file2 = str(l) + '.csv'
        raw1 = load(path1, file1, False)
        raw2 = load(path2, file2, False)
        buff1 = raw1.pop(target_depedencies[0])
        buff2 = raw2.pop(target_depedencies[1])
        res = []
        if compute_target_constrain:
            constr_velo1 = []
            constr_velo2 = []
            for i in range(len(index)):
                constr_velo1.append(buff1[index[i]])
                constr_velo2.append(buff2[index[i]])
                    
            aver_velo1 = np.mean(constr_velo1)
            aver_velo2 = np.mean(constr_velo2)

            velo1_prime = []
            velo2_prime = []
            for i in range(len(index)):
                val = constr_velo1[i] - aver_velo1
                velo1_prime.append(val)
                val = constr_velo2[i] - aver_velo2
                velo2_prime.append(val)

            product = []
            for i in range(len(index)):
                val = velo1_prime[i] * velo2_prime[i]
                product.append(val)

            res.append(np.mean(product))
        else:
            aver_velo1 = np.mean(buff1)
            aver_velo2 = np.mean(buff2)

            velo1_prime = []
            velo2_prime = []
            for i in range(len(buff1)):
                val = buff1[i] - aver_velo1
                velo1_prime.append(val)
                val = buff2[i] - aver_velo2
                velo2_prime.append(val)

            product = []
            for i in range(len(buff1)):
                val = velo1_prime[i] * velo2_prime[i]
                product.append(val)

            res.append(np.mean(product))

        print("Computed turbulent momentum flux at layer ", l, ": ", res[0])            
        label = 'layer_' + str(l)
        non_scaled_at_layer.insert(loc=l, column=label, value=res)
