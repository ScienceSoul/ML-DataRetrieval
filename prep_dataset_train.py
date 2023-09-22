import numpy as np

from global_defs import *
from HAGIBIS_params import *

def loop_common_3(non_scaled_at_layer, non_scaled_layers_files, num_vert_layers, tree_from_glogal_grid, t,
                  store_indexes=None, num_training_samples=None):
    if compute_target:
        if target == 'TURB_MOMENTUM_FLUXES':
            compute_turb_moment_fluxes(non_scaled_at_layer, num_vert_layers, path_to_V_non_scaled, time_stamps, t)
        else:
            print("Computed target not available.")
            sys.exit()
    else:    
        for l in range(num_vert_layers):
            path = path_to_V_non_scaled + time_stamps[t] + '/' + target
            raw_data_non_scaled = load(path, non_scaled_layers_files[l], False)
            buffer = raw_data_non_scaled.pop(target)
            if tree_from_glogal_grid:
                label = 'layer_' + str(l)
                non_scaled_at_layer.insert(loc=l, column=label, value=buffer)
            else:
                list_of_values = []
                for i in range(num_training_samples):
                    list_of_values.append(buffer[store_indexes[i]])
                label = 'layer_' + str(l)
                non_scaled_at_layer.insert(loc=l, column=label, value=list_of_values)


def create_train_dataset_zero_order(NN_inputs, eta, num_training_time_stamps, time_stamp_sampling_times,
                               num_training_samples, non_scaled_layers_files, num_columns):
    ii = 0
    first_time = True
    for t in range(num_training_time_stamps):

        if ignore_timestamp:
            if t == ignored_timestamp:
                continue

        Data_holder = pd.DataFrame()
        X, Y, raw_data = loop_common_1(Data_holder, t, path_to_V, time_stamps, first_time)

        # If the kd-tree is constructed from the global grid,
        # construct only once since the horizontal coordinates never change
        if compute_kdTree_from_global_grid:
            if first_time:
                NeighborsTree = loop_common_2(X, Y, False)
                first_time = False

        for sp in range(time_stamp_sampling_times):

            sample_data = raw_data.sample(n=num_training_samples)
            if not compute_kdTree_from_global_grid:
                store_indexes = []
                for i in range(num_training_samples):
                    store_indexes.append(sample_data.index[i])

            sample_data = sample_data.reset_index(drop=True)

            X_sample = sample_data[data_x]
            Y_sample = sample_data[data_y]

            # Get the non scaled values at each layer
            Non_scaled_at_layer = pd.DataFrame()
            if compute_kdTree_from_global_grid:
                loop_common_3(Non_scaled_at_layer, non_scaled_layers_files, num_vert_layers, True, t)
            else:
                loop_common_3(Non_scaled_at_layer, non_scaled_layers_files, num_vert_layers, False, t,
                              store_indexes=store_indexes, num_training_samples=num_training_samples)


            coord_sample = [[0 for j in range(2)] for i in range(num_training_samples)]
            for i in range(num_training_samples):
                coord_sample[i][0] = X_sample[i]
                coord_sample[i][1] = Y_sample[i]

            
            # If constrained, discard the previous sampling and do again
            if constrained_target:
                if not compute_kdTree_from_global_grid:
                    print("Constrained mode not supported when the kdTree is not computed on the global grid.")

                constrained_buffer = get_sample_with_constrain(raw_data, Non_scaled_at_layer, LOWER_LIMIT)
                if constrained_buffer.size < num_training_samples:
                    print("Number of contrained values smaller than the size of the training sample.")
                    sys.exit() 

                constrained_sample = constrained_buffer.sample(n=num_training_samples)
                constrained_sample = constrained_sample.reset_index(drop=True)
                X = raw_data[data_x]
                Y = raw_data[data_y]
                index = constrained_sample['INDEX']
                for i in range(num_training_samples):
                    coord_sample[i][0] = X[index[i]]
                    coord_sample[i][1] = Y[index[i]]


            # If the kd-tree is constructed from the sampled data
            if not compute_kdTree_from_global_grid:
                NeighborsTree = spatial.cKDTree(coord_sample, leafsize=100)

            for item in coord_sample:
                dd, loc = NeighborsTree.query(item, k=required_neighbors + 1)
                print("loc:", item)

                for l in range(num_vert_layers):
                    # The non scaled value is used for the label
                    label = 'layer_' + str(l)
                    var_non_scaled = Non_scaled_at_layer[label]

                    NN_inputs[ii][0] = var_non_scaled[loc[0]]

                    var_idx = 0
                    jj = 1
                    for vv in V_dict.values():
                        if vv == 1:
                            var = Data_holder[V[var_idx]]
                            idx = 1
                            for j in range(0, required_neighbors):
                                if compute_kdTree_from_global_grid:
                                    NN_inputs[ii][jj] = var[loc[idx]]
                                else:
                                    NN_inputs[ii][jj] = var[store_indexes[loc[idx]]]
                                jj = jj + 1
                                idx = idx + 1
                        var_idx = var_idx + 1

                    loop_common_4(NN_inputs, eta, dd, ii, jj, num_columns, l, False)
                    ii = ii + 1


def create_train_dataset_first_order(NN_inputs, eta, num_training_time_stamps, time_stamp_sampling_times,
                               num_training_samples, non_scaled_layers_files, num_columns, lat=None, long=None):

    ii = 0
    first_time = True

    for t in range(num_training_time_stamps):

        if ignore_timestamp:
            if t == ignored_timestamp:
                continue

        Data_holder = pd.DataFrame()
        X, Y, raw_data = loop_common_1(Data_holder, t, path_to_V, time_stamps, first_time)

        # If the kd-tree is constructed from the global grid,
        # construct only once since the horizontal coordinates never change
        if compute_kdTree_from_global_grid:
            if first_time:
                NeighborsTree = loop_common_2(X, Y, False)
                first_time = False


        sample_data = raw_data.sample(n=num_training_samples)
        if not compute_kdTree_from_global_grid:
            store_indexes = []
            for i in range(num_training_samples):
                store_indexes.append(sample_data.index[i])

        sample_data = sample_data.reset_index(drop=True)

        X_sample = sample_data[data_x]
        Y_sample = sample_data[data_y]

        # Get the non scaled values at each layer
        Non_scaled_at_layer = pd.DataFrame()
        if compute_kdTree_from_global_grid:
            loop_common_3(Non_scaled_at_layer, non_scaled_layers_files, num_vert_layers, True, t)
        else:
            loop_common_3(Non_scaled_at_layer, non_scaled_layers_files, num_vert_layers, False, t,
                              store_indexes=store_indexes, num_training_samples=num_training_samples)


        coord_sample = [[0 for j in range(2)] for i in range(num_training_samples)]
        for i in range(num_training_samples):
            coord_sample[i][0] = X_sample[i]
            coord_sample[i][1] = Y_sample[i]


        # If constrained, discard the previous sampling and do again
        if constrained_target or compute_target_constrain:
            if not compute_kdTree_from_global_grid:
                 print("Constrained mode not supported when the kdTree is not computed on the global grid.")

            if compute_target:
                constrained_buffer = get_sample_with_constrain(raw_data, Non_scaled_at_layer, compute_target_lower_limit, 
                                     path_to_V_non_scaled=path_to_V_non_scaled, time_stamps=time_stamps, t=t)
            else:
                if compute_target_constrain:
                    print("Constrain the computed target but the target is not computed.")
                    sys.exit()

                constrained_buffer = get_sample_with_constrain(raw_data, Non_scaled_at_layer, LOWER_LIMIT)
            if constrained_buffer.size < num_training_samples:
                print("Number of contrained values smaller than the size of the training sample.")
                sys.exit() 

            constrained_sample = constrained_buffer.sample(n=num_training_samples)
            constrained_sample = constrained_sample.reset_index(drop=True)
            X = raw_data[data_x]
            Y = raw_data[data_y]
            index = constrained_sample['INDEX']
            for i in range(num_training_samples):
                coord_sample[i][0] = X[index[i]]
                coord_sample[i][1] = Y[index[i]]


        # If the kd-tree is constructed from the sampled data
        if not compute_kdTree_from_global_grid:
            NeighborsTree = spatial.cKDTree(coord_sample, leafsize=100)

        num_neighbors = required_neighbors
        taken_nodes = required_neighbors
        for incr in range(0, number_grid_points_apart):

            for item in coord_sample:
                far_dd, far_loc = NeighborsTree.query(item, k=num_neighbors + 1)
                print("loc:", item)

                dd = np.zeros(taken_nodes+1)
                loc = np.zeros(taken_nodes+1)
                dd[0] = far_dd[0]
                loc[0] = far_loc[0]

                max_idx = np.argmax(far_dd)
                st = max_idx - taken_nodes + 1
                for i in range(1, taken_nodes+1):
                    dd[i] = far_dd[st]
                    loc[i] = far_loc[st]
                    st = st + 1

                # Randomly pick required_neighbors indexes
                rand_idx = random.sample(range(1,dd.shape[0]), required_neighbors)

                idx = 1
                for j in range(0, required_neighbors):
                    for l in range(num_vert_layers):
                        col = 0
                        # The non scaled value is used for the label
                        label = 'layer_' + str(l)
                        var_non_scaled = Non_scaled_at_layer[label]

                        if first_order_learn_mode == 1 or first_order_learn_mode == 3:
                            if len(var_non_scaled) > 1:
                                NN_inputs[ii][col] = var_non_scaled[loc[rand_idx[j]]]
                            else:
                                NN_inputs[ii][col] = var_non_scaled[0]
                        elif first_order_learn_mode == 2:
                            if target == 'TURB_MOMENTUM_FLUXES':
                                print("This mode does not make sense when the target is the turbulent momentum flux.")
                                sys.exit()
                            NN_inputs[ii][col] = var_non_scaled[loc[rand_idx[j]]] - var_non_scaled[loc[0]]

                        col = col + 1

                        var_idx = 0
                        for vv in V_dict.values():
                            if vv == 1:
                                var = Data_holder[V[var_idx]]
                                if compute_kdTree_from_global_grid:
                                    if V[var_idx] == 'COR_NORTH' or V[var_idx] == 'COR_EAST' \
                                            or V[var_idx] == 'DRY_MASS':
                                        NN_inputs[ii][col] = var[loc[0]]
                                    else:
                                        if first_order_learn_mode == 1 or first_order_learn_mode == 2:
                                            NN_inputs[ii][col] = var[loc[rand_idx[j]]] - var[loc[0]]
                                        elif first_order_learn_mode == 3:
                                            if V[var_idx] == target:
                                                NN_inputs[ii][col] = var[loc[0]]
                                                col = col + 1
                                                NN_inputs[ii][col] = var[loc[rand_idx[j]]] - var[loc[0]]
                                            else:
                                                NN_inputs[ii][col] = var[loc[rand_idx[j]]] - var[loc[0]]
                                else:
                                    # TODO: higher order learning not implemented yet
                                    NN_inputs[ii][col] = var[store_indexes[loc[rand_idx[j]]]] - \
                                                         var[store_indexes[loc[0]]]
                                col = col + 1
                            var_idx = var_idx + 1

                        lon1 = long[loc[0]]
                        lat1 = lat[loc[0]]
                        lon2 = long[loc[rand_idx[j]]]
                        lat2 = lat[loc[rand_idx[j]]]
                        bearing = compute_bearing(lat1, lon1, lat2, lon2)
                        compass_val = compass(bearing)
                        loop_common_4(NN_inputs, eta, dd, ii, col, num_columns, l, False,
                                          neighbor_idx=rand_idx[j], compass=compass_val)
                        ii = ii + 1
                    idx = idx + 1

            taken_nodes = taken_nodes + 8
            num_neighbors = num_neighbors + taken_nodes
