from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import random

from prep_dataset_test import *

num_test_samples = 1 
num_vert_profiles = 5

# Compute and draw 2D map(s)
compute_draw_2D_map = False
scale_to_truth = True
compute_draw_2D_map_diffs = False
first_2D_map_layer_index = 1
last_2D_map_layer_index = 3
select_time_stamp = -1

# Use log scale for y axis
use_log = False

# Draw an additional plot 
draw_additional_plot = False
additional_plot_variable = ""
add_variable_label = ""

# Draw profiles of difference between NN and WRF
draw_profiles_diff = True

num_test_time_stamps = len(time_stamps)

eta = get_eta(num_vert_layers, path_to_eta)
print(eta)

non_scaled_layers_files = get_non_scaled_layers_files(num_vert_layers)

# Compute the number of inputs to the network
num_columns = compute_num_columns()

NN_columns = define_nn_columns()

if order_of_learning == 1:
    if compute_target:
        # Just pick a variable to read X and Y if the target is computed
        path = path_to_V_non_scaled + time_stamps[0] + '/' + key1
    else:
        path = path_to_V_non_scaled + time_stamps[0] + '/' + target

    temp = load(path, non_scaled_layers_files[0], False)
    temp = temp.dropna()
    X_non_scaled = temp[data_x]
    Y_non_scaled = temp[data_y]

def draw_2d_plot(x, y, z, title, block, cmap, scale_cbar=None, labels=None, diff=None):
    plt.figure(figsize=(8, 5))
    if diff:
        zz = abs(z - labels)
    else:
        zz = z
    var_plt = plt.scatter(x, y, c=zz, cmap=cmap)

    if scale_cbar:
        var_plt.set_clim(np.min(labels), np.max(labels))

    if diff:
        if target == 'U' or target == 'V':
            var_plt.set_clim(0, 20)
        elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
            var_plt.set_clim(0, 1)
        else:
            var_plt.set_clim(0, 0.04)

    plt.title(title)
    plt.xlabel('Longitude (degrees east)', fontsize=13)
    plt.ylabel('Latitude (degrees north)', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    cbar = plt.colorbar(var_plt)
    cbar.ax.tick_params(labelsize=12)
    plt.show(block=block)


def test_neural_network_zero_order(compute_2d_map, num_vert_profiles=None, map_layer=None, last_layer=None,
                                   time_stamp=None):

    if compute_2d_map:
        print('----- Compute map at layer: ', map_layer)

    num_vert = 0
    if not compute_2d_map:
        num_vert = num_vert_profiles
        if num_test_samples > 1:
            num_vert = 1

    first_time = True
    if compute_2d_map:
        start = 0
        end = 1
    else:
        start = 0
        end = num_vert

    for p in range(start, end):

        if compute_2d_map:
            t = time_stamp
        else:
            t = random.randint(0, num_test_time_stamps-1)

        print('----- Data sampled using time stamp:')
        print(time_stamps[t])

        if compute_2d_map:
            NN_inputs, longi, lat = create_test_dataset_zero_order(t, compute_2d_map, num_columns, eta,
                                                non_scaled_layers_files, first_time, map_layer=map_layer)
        else:
            NN_inputs, longi, lat = create_test_dataset_zero_order(t, compute_2d_map, num_columns, eta,
                                                non_scaled_layers_files, first_time,
                                                num_test_samples=num_test_samples)


        nn_gen_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
        print(nn_gen_dataset.tail())

        v1_gen_set = pd.DataFrame()
        v2_gen_set = pd.DataFrame()
        v3_gen_set = pd.DataFrame()
        v4_gen_set = pd.DataFrame()
        v5_gen_set = pd.DataFrame()
        v6_gen_set = pd.DataFrame()
        d_gen_set = pd.DataFrame()
        eta_gen_set = pd.DataFrame()

        final_nn_inputs(nn_gen_dataset, required_neighbors, v1_gen_set, v2_gen_set, v3_gen_set, v4_gen_set,
                        v5_gen_set, v6_gen_set, d_gen_set, eta_gen_set, False)

        gen_labels = nn_gen_dataset.pop('TAR')

        print('gen labels:')
        print(gen_labels.tail())

        vec_gen = []
        var_idx = 0
        for item in V_dict.values():
            if item == 1:
                if (var_idx + 1) == 1:
                    vec_gen.append(v1_gen_set)
                if (var_idx + 1) == 2:
                    vec_gen.append(v2_gen_set)
                if (var_idx + 1) == 3:
                    vec_gen.append(v3_gen_set)
                if (var_idx + 1) == 4:
                    vec_gen.append(v4_gen_set)
                if (var_idx + 1) == 5:
                    vec_gen.append(v5_gen_set)
                if (var_idx + 1) == 6:
                    vec_gen.append(v6_gen_set)
            var_idx = var_idx + 1

        if include_distance:
            vec_gen.append(d_gen_set)

        vec_gen.append(eta_gen_set)

        compute_gen = loaded_model.predict(vec_gen).flatten()

        if not compute_2d_map:
            error = compute_gen - gen_labels

            if num_test_samples > 1:
                plt.figure()
                plt.scatter(gen_labels, compute_gen)
                if target == 'U' or target == 'V' or target == 'W':
                    plt.xlabel('True values [m/s]')
                    plt.ylabel('Predictions [m/s]')
                elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
                    plt.xlabel(r'$\mathrm{True\,values}\,[k s^{-1}]$')
                    plt.ylabel(r'$\mathrm{Predictions}\,[k s^{-1}]$')
                plt.axis('equal')
                plt.axis('square')
                if target == 'U' or target == 'V' or target == 'W':
                    plt.xlim([-50, plt.xlim()[1]])
                    plt.ylim([-50, plt.ylim()[1]])
                    _ = plt.plot([-100, 100], [-100, 100])
                elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
                    plt.xlim([-plt.xlim()[1], plt.xlim()[1]])
                    plt.ylim([-plt.ylim()[1], plt.ylim()[1]])
                    _ = plt.plot([-10, 10], [-10, 10])

                plt.show(block=False)

                plt.figure()
                plt.hist(error, bins=25)
                if target == 'U' or target == 'V' or target == 'W':
                    plt.xlabel('Prediction error [m/s]')
                elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
                      plt.xlabel(r'$\mathrm{Prediction\,error}\,[k s^{-1}]$')

                _ = plt.ylabel("Count")
                plt.show()
            else:
                plt.figure()
                plt.gca().invert_yaxis()
                plt.plot(gen_labels, eta, 'b', label='WRF')
                plt.plot(compute_gen, eta, 'r', label='NN')
                plt.yscale('log')
                if target == 'U' or target == 'V' or target == 'W':
                    plt.xlabel('V (m/s)')
                elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
                    plt.xlabel(r'$\xi\,(k s^{-1})$')
                plt.ylabel(r'$\eta$')
                plt.legend(loc='lower left')
                if p == (num_vert - 1):
                    plt.show()
                else:
                    plt.show(block=False)
        else:
            # The reference 2D distribution from the forecasting model
            if target == 'U':
                title_variable = 'x-wind component'
                cmap = 'seismic'
            elif target == 'V':
                title_variable = 'y-wind component'
                cmap = 'seismic'
            elif target == 'QVAPOR':
                title_variable = 'water vapor'
                cmap = 'RdBu'
            elif target == 'REL_VERT_VORT':
                title_variable = 'relative vorticity'
                cmap = 'seismic'
            elif target == 'ABS_VERT_VORT':
                title_variable = 'absolute vorticity'
                cmap = 'seismic'
            else:
                title_variable = 'undefined'
                cmap = 'seismic'

            title_eta_label = '(' + '$\eta$' + '=' + str(eta[map_layer]) + ')'
            title = 'WRF ' + title_variable + ' at layer ' + str(map_layer) + ' ' + title_eta_label
            draw_2d_plot(longi, lat, gen_labels, title, False, cmap)

            # The 2D distribution computed with the neural network model
            title = 'NN ' + title_variable + ' at layer ' + str(map_layer) + ' ' + title_eta_label
            if compute_draw_2D_map_diffs:
                block = False
            else:
                if map_layer == (last_layer - 1):
                    block = True
                else:
                    block = False

            draw_2d_plot(longi, lat, compute_gen, title, block, cmap, scale_cbar=scale_to_truth, labels=gen_labels)

            if compute_draw_2D_map_diffs:
                title = 'Absolute error on ' + title_variable + ' at layer ' + str(map_layer) \
                        + ' ' + title_eta_label
                block = False
                if map_layer == (last_layer - 1):
                    block = True

                draw_2d_plot(longi, lat, compute_gen, title, block, 'jet', labels=gen_labels, diff=True)


def set_xlabel(diff_plot):
    if target == 'U' or target == 'V' or target == 'W':
        if diff_plot:
            plt.xlabel('Abs. diff. (m/s)')
        else:
            if target == 'U': 
                plt.xlabel('U (m/s)')
            elif target == 'V':
                plt.xlabel('V (m/s)')
            else:
                plt.xlabel('W (m/s)')
    elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
        if diff_plot:
            plt.xlabel(r'$\mathrm{Abs. diff.}\,(k s^{-1})$')
        else:
            plt.xlabel(r'$\xi\,(k s^{-1})$')
    elif target == 'TURB_MOMENTUM_FLUXES':
        if diff_plot:
            plt.xlabel(r'$\mathrm{Abs. diff.}\,(m^{2} s^{-2})$')
        else:
            plt.xlabel(r'$\mathrm{Turb. Moment. Flux}\,(m^{2} s^{-2})$')
    else:
        plt.xlabel('var ()')


def line_plot(x1, y, title, block, x2=None, x3=None, xlabel=None, y_log=None):
    plt.figure()
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.plot(x1, eta, 'b', label='WRF')
    if x2:
        plt.plot(x2, eta, 'r', label='NN')
    if x3:
        plt.plot(x3, eta, 'g', label='SELF')

    if y_log:
        if y_log == True:
            plt.yscale('log')

    if xlabel:
        plt.xlabel(xlabel)
    else:
       set_xlabel(False)

    plt.ylabel(r'$\eta$')
    plt.legend(loc='lower left')
    if block == False:
        plt.show(block=False)
    else:
        plt.show()


def get_node_with_all_directions(tree, raw_data, non_scaled_at_layer=None, t=None):
    while True:
        
        coord_sample = [[0 for j in range(2)] for i in range(1)]

        if constrained_target or compute_target_constrain:
            if compute_target:
                # non_scaled_at_layer is actually not used in this case
                constrained_buffer = get_sample_with_constrain(raw_data, non_scaled_at_layer, compute_target_lower_limit, 
                                     path_to_V_non_scaled=path_to_V_non_scaled, time_stamps=time_stamps, t=t)
            else:
                constrained_buffer = get_sample_with_constrain(raw_data, non_scaled_at_layer, LOWER_LIMIT)

            constrained_sample = constrained_buffer.sample(n=1)
            constrained_sample = constrained_sample.reset_index(drop=True)
            X = raw_data[data_x]
            Y = raw_data[data_y]
            index = constrained_sample['INDEX']
            coord_sample[0][0] = X[index[0]]
            coord_sample[0][1] = Y[index[0]]
        else:
            sample_data = raw_data.sample(n=1)
            sample_data = sample_data.reset_index(drop=True)
            print(sample_data.tail())

            x_sample = sample_data[data_x]
            y_sample = sample_data[data_y]

            for i in range(1):
                coord_sample[i][0] = x_sample[i]
                coord_sample[i][1] = y_sample[i]

        for item in coord_sample:
            dd, loc = tree.query(item, k=required_neighbors + 1)
            print("loc:", item)

        north = False
        north_east = False
        east = False
        south_east = False
        south = False
        south_west = False
        west = False
        north_west = False

        idx = 1
        for j in range(0, required_neighbors):

            lon1 = X_non_scaled[loc[0]]
            lat1 = Y_non_scaled[loc[0]]
            lon2 = X_non_scaled[loc[idx]]
            lat2 = Y_non_scaled[loc[idx]]
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            compass_val = compass(bearing)

            if compass_val == 0:
                north = True
            elif compass_val == 1:
                north_east = True
            elif compass_val == 2:
                east = True
            elif compass_val == 3:
                south_east = True
            elif compass_val == 4:
                south = True
            elif compass_val == 5:
                south_west = True
            elif compass_val == 6:
                west = True
            elif  compass_val == 7:
                north_west = True

            idx = idx + 1

        if north and north_east and east and south_east and \
            south and south_west and west and north_west:
            break

    return dd, loc


def get_node_and_furthest(tree, raw_data, number_grid_points_apart_test, non_scaled_at_layer=None, t=None):

    coord_sample = [[0 for j in range(2)] for i in range(1)]
    if constrained_target or compute_target_constrain:
        if compute_target:
            # non_scaled_at_layer is actually not used in this case
            constrained_buffer = get_sample_with_constrain(raw_data, non_scaled_at_layer, compute_target_lower_limit, 
                                     path_to_V_non_scaled=path_to_V_non_scaled, time_stamps=time_stamps, t=t)
        else:   
            constrained_buffer = get_sample_with_constrain(raw_data, non_scaled_at_layer, LOWER_LIMIT)

        constrained_sample = constrained_buffer.sample(n=1)
        constrained_sample = constrained_sample.reset_index(drop=True)
        X = raw_data[data_x]
        Y = raw_data[data_y]
        index = constrained_sample['INDEX']
        coord_sample[0][0] = X[index[0]]
        coord_sample[0][1] = Y[index[0]]
    else:
        sample_data = raw_data.sample(n=1)
        sample_data = sample_data.reset_index(drop=True)
        print(sample_data.tail())
    
        x_sample = sample_data[data_x]
        y_sample = sample_data[data_y]

        for i in range(1):
            coord_sample[i][0] = x_sample[i]
            coord_sample[i][1] = y_sample[i]

    num_neighbors = required_neighbors
    taken_nodes = required_neighbors
    for incr in range(0, number_grid_points_apart_test):
        for item in coord_sample:
            far_dd, far_loc = tree.query(item, k=num_neighbors + 1)
            print("loc:", item)

            dd = np.zeros(taken_nodes + 1)
            loc = np.zeros(taken_nodes + 1)
            dd[0] = far_dd[0]
            loc[0] = far_loc[0]

            max_idx = np.argmax(far_dd)
            st = max_idx - taken_nodes + 1
            for i in range(1, taken_nodes + 1):
                dd[i] = far_dd[st]
                loc[i] = far_loc[st]
                st = st + 1

        taken_nodes = taken_nodes + 8
        num_neighbors = num_neighbors + taken_nodes

    idx = random.sample(range(1, dd.shape[0]), 1)
    return idx[0], dd, loc


def first_order_test_mode_1(tree, raw_data, data_holder, non_scaled_at_layer, t):

    if constrained_target or compute_target_constrain:
        if compute_target:
            dd, loc = get_node_with_all_directions(tree, raw_data, non_scaled_at_layer=non_scaled_at_layer, t=t)
        else:
            if compute_target_constrain:
                print("Constrain the computed target but the target is not computed.")
                sys.exit()

            dd, loc = get_node_with_all_directions(tree, raw_data, non_scaled_at_layer=non_scaled_at_layer)
    else:
        dd, loc = get_node_with_all_directions(tree, raw_data)

    NN_inputs = [[0 for j in range(num_columns)] for i in range(1)]

    N_neighbors_WRF = []
    N_neighbors_NN = []

    NE_neighbors_WRF = []
    NE_neighbors_NN = []

    E_neighbors_WRF = []
    E_neighbors_NN = []

    SE_neighbors_WRF = []
    SE_neighbors_NN = []

    S_neighbors_WRF = []
    S_neighbors_NN = []

    SW_neighbors_WRF = []
    SW_neighbors_NN = []

    W_neighbors_WRF = []
    W_neighbors_NN = []

    NW_neighbors_WRF = []
    NW_neighbors_NN = []

    if draw_additional_plot:
        N_add_var = []
        NE_add_var = []
        E_add_var = []
        SE_add_var = []
        S_add_var = []
        SW_add_var = []
        W_add_var = []
        NW_add_var = []

    if draw_profiles_diff:
        N_diff = []
        NE_diff = []
        E_diff = []
        SE_diff = []
        S_diff = []
        SW_diff = []
        W_diff = []
        NW_diff = []

        
    for l in range(0, num_vert_layers):
        print('>>>>>>>> layer: ', l)
        label = 'layer_' + str(l)
        var_non_scaled = non_scaled_at_layer[label]
        idx = 1
        for j in range(0, required_neighbors):
            col = 0
            if first_order_learn_mode == 1 or first_order_learn_mode == 3:
                if len(var_non_scaled) > 1:
                    NN_inputs[0][col] = var_non_scaled[loc[idx]]
                else:
                    NN_inputs[0][col] = var_non_scaled[0]
            elif first_order_learn_mode == 2:
                if target == 'TURB_MOMENTUM_FLUXES':
                    print("This mode does not make sense when the target is the turbulent momentum flux.")
                    sys.exit()
                NN_inputs[0][col] = var_non_scaled[loc[idx]] - var_non_scaled[loc[0]]

            col = col + 1

            var_idx = 0
            for vv in V_dict.values():
                if vv == 1:
                    var = data_holder[V[var_idx]]
                    if V[var_idx] == 'COR_NORTH' or V[var_idx] == 'COR_EAST' \
                            or V[var_idx] == 'DRY_MASS':
                        NN_inputs[0][col] = var[loc[0]]
                    else:
                        if first_order_learn_mode == 1 or first_order_learn_mode == 2:
                            NN_inputs[0][col] = var[loc[idx]] - var[loc[0]]
                        elif first_order_learn_mode == 3:
                            if V[var_idx] == target:
                                NN_inputs[0][col] = var[loc[0]]
                                col = col + 1
                                NN_inputs[0][col] = var[loc[idx]] - var[loc[0]]
                            else:
                                NN_inputs[0][col] = var[loc[idx]] - var[loc[0]]
                    col = col + 1
                var_idx = var_idx + 1

            lon1 = X_non_scaled[loc[0]]
            lat1 = Y_non_scaled[loc[0]]
            lon2 = X_non_scaled[loc[idx]]
            lat2 = Y_non_scaled[loc[idx]]
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            compass_val = compass(bearing)
            loop_common_4(NN_inputs, eta, dd, 0, col, num_columns, l, True,
                          neighbor_idx=idx, compass=compass_val, compute_2d_map=False)

            nn_gen_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
            print(nn_gen_dataset.tail())

            gen_labels = nn_gen_dataset.pop('TAR')
            print('gen labels:')
            print(gen_labels.tail())

            compute_gen = loaded_model.predict(nn_gen_dataset).flatten()
            print('+++++++ NN computed value: ', compute_gen[0])
            if len(var_non_scaled) > 1:
                print('+++++++ [self]->[neigh.]:', var_non_scaled[loc[0]], var_non_scaled[loc[idx]])
            else:
                print('+++++++ [self]->[neigh.]:', var_non_scaled[0])

            if first_order_learn_mode == 1 or first_order_learn_mode == 3:
                value = compute_gen[0]
            elif first_order_learn_mode == 2:
                value = var_non_scaled[loc[0]] + compute_gen[0]

            if draw_profiles_diff:
                if len(var_non_scaled) > 1:
                    diff = abs(compute_gen[0] - var_non_scaled[loc[idx]])
                else:
                    diff = abs(compute_gen[0] - var_non_scaled[0])

            if draw_additional_plot:
                path = path_to_V_non_scaled + time_stamps[t] + '/' + additional_plot_variable
                if additional_plot_variable == 'U' or additional_plot_variable == 'V':
                    file = str(l) + '_interpol.csv'
                else: 
                    file = str(l) + '.csv'
                raw = load(path, file, False)
                add_var = raw.pop(additional_plot_variable)

            if compass_val == 0:
                print('Towards North')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    N_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    N_neighbors_WRF.append(var_non_scaled[0])
                N_neighbors_NN.append(value)
                if draw_additional_plot:
                    N_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    N_diff.append(diff)
            elif compass_val == 1:
                print('Towards North-East')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    NE_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    NE_neighbors_WRF.append(var_non_scaled[0])
                NE_neighbors_NN.append(value)
                if draw_additional_plot:
                    NE_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    NE_diff.append(diff)
            elif compass_val == 2:
                print('Towards East')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    E_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    E_neighbors_WRF.append(var_non_scaled[0])
                E_neighbors_NN.append(value)
                if draw_additional_plot:
                    E_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    E_diff.append(diff)
            elif compass_val == 3:
                print('Towards South-East')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    SE_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    SE_neighbors_WRF.append(var_non_scaled[0])
                SE_neighbors_NN.append(value)
                if draw_additional_plot:
                    SE_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    SE_diff.append(diff)
            elif compass_val == 4:
                print('Towards South')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    S_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    S_neighbors_WRF.append(var_non_scaled[0])
                S_neighbors_NN.append(value)
                if draw_additional_plot:
                    S_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    S_diff.append(diff)
            elif compass_val == 5:
                print('Towards South-West')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    SW_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    SW_neighbors_WRF.append(var_non_scaled[0])
                SW_neighbors_NN.append(value)
                if draw_additional_plot:
                    SW_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    SW_diff.append(diff)
            elif compass_val == 6:
                print('Towards West')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    W_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    W_neighbors_WRF.append(var_non_scaled[0])
                W_neighbors_NN.append(value)
                if draw_additional_plot:
                    W_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    W_diff.append(diff)
            elif compass_val == 7:
                print('Towards North-West')
                print('--------------------')
                if len(var_non_scaled) > 1:
                    NW_neighbors_WRF.append(var_non_scaled[loc[idx]])
                else:
                    NW_neighbors_WRF.append(var_non_scaled[0])
                NW_neighbors_NN.append(value)
                if draw_additional_plot:
                    NW_add_var.append(add_var[loc[idx]])
                if draw_profiles_diff:
                    NW_diff.append(diff)

            idx = idx + 1

    for i in range(0, 8):
        if i == 0:
            if len(N_neighbors_WRF) != 0 and len(N_neighbors_NN) != 0:
                line_plot(N_neighbors_WRF, eta, 'Towards North', False, x2=N_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(N_add_var, eta, 'Towards North', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 1:
            if len(NE_neighbors_WRF) != 0 and len(NE_neighbors_NN) != 0:
                line_plot(NE_neighbors_WRF, eta, 'Towards North-East', False, x2=NE_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(NE_add_var, eta, 'Towards North-East', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 2:
            if len(E_neighbors_WRF) != 0 and len(E_neighbors_NN) != 0:
                line_plot(E_neighbors_WRF, eta, 'Towards East', False, x2=E_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(E_add_var, eta, 'Towards East', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 3:
            if len(SE_neighbors_WRF) != 0 and len(SE_neighbors_NN) != 0:
                line_plot(SE_neighbors_WRF, eta, 'Towards South-East', False, x2=SE_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(SE_add_var, eta, 'Towards South-East', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 4:
            if len(S_neighbors_WRF) != 0 and len(S_neighbors_NN) != 0:
                line_plot(S_neighbors_WRF, eta, 'Towards South', False, x2=S_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(S_add_var, eta, 'Towards South', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 5:
            if len(SW_neighbors_WRF) != 0 and len(SW_neighbors_NN) != 0:
                line_plot(SW_neighbors_WRF, eta, 'Towards South-West', False, x2=SW_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(SW_add_var, eta, 'Towards South-West', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 6:
            if len(W_neighbors_WRF) != 0 and len(W_neighbors_NN) != 0:
                line_plot(W_neighbors_WRF, eta, 'Towards West', False, x2=W_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                line_plot(W_add_var, eta, 'Towards West', False, xlabel=add_variable_label, y_log=use_log)
        elif i == 7:
            if len(NW_neighbors_WRF) != 0 and len(NW_neighbors_NN) != 0:
                if draw_profiles_diff or draw_additional_plot:
                    block = False
                else:
                    block = True
                line_plot(NW_neighbors_WRF, eta, 'Towards North-West', block, x2=NW_neighbors_NN, y_log=use_log)
            if draw_additional_plot:
                if draw_profiles_diff:
                    block = False
                else:
                    block = True

                line_plot(NW_add_var, eta, 'Towards North-West', block, xlabel=add_variable_label, y_log=use_log)

    if draw_profiles_diff:
        diff_profiles = [[0 for x in range(num_vert_layers)] for x in range(8)]
        legends = []
        for i in range(0, 8):
            if i == 0:
                legends.append('North')
            elif i == 1:
                legends.append('North-East')
            elif i == 2:
                legends.append('East')
            elif i == 3:
                legends.append('South-East')
            elif i == 4:
                legends.append('South')
            elif i == 5:
                legends.append('South-West')
            elif i == 6:
                legends.append('West')
            elif i == 7:
                legends.append('North-West')

            for j in range(0, num_vert_layers):
                if i == 0:
                    diff_profiles[i][j] = N_diff[j]
                elif i == 1:
                    diff_profiles[i][j] = NE_diff[j]
                elif i == 2:
                    diff_profiles[i][j] = E_diff[j]
                elif i == 3:
                    diff_profiles[i][j] = SE_diff[j]
                elif i == 4:
                    diff_profiles[i][j] = S_diff[j]
                elif i == 5:
                    diff_profiles[i][j] = SW_diff[j]
                elif i == 6:
                    diff_profiles[i][j] = W_diff[j]
                elif i == 7:
                    diff_profiles[i][j] = NW_diff[j]

        plt.figure()
        plt.gca().invert_yaxis()
        plt.title('Abs. diff.')
        for i in range(0, 8):
            plt.plot(diff_profiles[i], eta, label=legends[i])

        if use_log:
            plt.yscale('log')   
        set_xlabel(True)
        plt.ylabel(r'$\eta$')
        plt.legend(loc='lower right')
        plt.show()


def first_order_test_mode_2(tree, raw_data, data_holder, non_scaled_at_layer, t):
    
    if draw_profiles_diff:
        diff_profiles = [[0 for x in range(num_vert_layers)] for x in range(6)]
        legends = []

    do_n_times = 6
    for nt in range(0, do_n_times):
        if constrained_target or compute_target_constrain:
            if compute_target:
                idx, dd, loc = get_node_and_furthest(tree, raw_data, 
                           number_grid_points_apart_test, non_scaled_at_layer=non_scaled_at_layer, t=t)    
            else:
                if compute_target_constrain:
                    print("Constrain the computed target but the target is not computed.")
                    sys.exit()

                idx, dd, loc = get_node_and_furthest(tree, raw_data, 
                           number_grid_points_apart_test, non_scaled_at_layer=non_scaled_at_layer)    
        else:
            idx, dd, loc = get_node_and_furthest(tree, raw_data, number_grid_points_apart_test)

        NN_inputs = [[0 for j in range(num_columns)] for i in range(1)]

        self_WRF = []
        furthest_WRF = []
        furthest_NN = []

        if draw_profiles_diff:
            diff_buff = []

        for l in range(0, num_vert_layers):
            print('>>>>>>>> layer: ', l)
            label = 'layer_' + str(l)
            var_non_scaled = non_scaled_at_layer[label]

            col = 0
            if first_order_learn_mode == 1 or first_order_learn_mode == 3:
                if len(var_non_scaled) > 1:
                    NN_inputs[0][col] = var_non_scaled[loc[idx]]
                else:
                    NN_inputs[0][col] = var_non_scaled[0]
            elif first_order_learn_mode == 2:
                if target == 'TURB_MOMENTUM_FLUXES':
                    print("This mode does not make sense when the target is the turbulent momentum flux.")
                    sys.exit()

                NN_inputs[0][col] = var_non_scaled[loc[idx]] - var_non_scaled[loc[0]]

            col = col + 1

            var_idx = 0
            for vv in V_dict.values():
                if vv == 1:
                    var = data_holder[V[var_idx]]
                    if V[var_idx] == 'COR_NORTH' or V[var_idx] == 'COR_EAST' \
                            or V[var_idx] == 'DRY_MASS':
                        NN_inputs[0][col] = var[loc[0]]
                    else:
                        if first_order_learn_mode == 1 or first_order_learn_mode == 2:
                            NN_inputs[0][col] = var[loc[idx]] - var[loc[0]]
                        elif first_order_learn_mode == 3:
                            if V[var_idx] == target:
                                NN_inputs[0][col] = var[loc[0]]
                                col = col + 1
                                NN_inputs[0][col] = var[loc[idx]] - var[loc[0]]
                            else:
                                NN_inputs[0][col] = var[loc[idx]] - var[loc[0]]
                    col = col + 1
                var_idx = var_idx + 1

            lon1 = X_non_scaled[loc[0]]
            lat1 = Y_non_scaled[loc[0]]
            lon2 = X_non_scaled[loc[idx]]
            lat2 = Y_non_scaled[loc[idx]]
            bearing = compute_bearing(lat1, lon1, lat2, lon2)
            compass_val = compass(bearing)
            loop_common_4(NN_inputs, eta, dd, 0, col, num_columns, l, True,
                          neighbor_idx=idx, compass=compass_val, compute_2d_map=False)

            nn_gen_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
            print(nn_gen_dataset.tail())

            gen_labels = nn_gen_dataset.pop('TAR')
            print('gen labels:')
            print(gen_labels.tail())

            compute_gen = loaded_model.predict(nn_gen_dataset).flatten()
            print('+++++++ NN computed value: ', compute_gen[0])
            if len(var_non_scaled) > 1:
                print('+++++++ [self]->[neigh.]:', var_non_scaled[loc[0]], var_non_scaled[loc[idx]])
            else:
                print('+++++++ [self]->[neigh.]:', var_non_scaled[0])

            if draw_profiles_diff:
                if len(var_non_scaled) > 1:
                    diff = abs(compute_gen[0] - var_non_scaled[loc[idx]])
                else:
                    diff = abs(compute_gen[0] - var_non_scaled[0])

            if l == 0:
                if compass_val == 0:
                    title = 'Towards North'
                    print(title)
                    print('--------------------')
                    legends.append('North')
                elif compass_val == 1:
                    title = 'Towards North-East'
                    print(title)
                    print('--------------------')
                    legends.append('North-East')
                elif compass_val == 2:
                    title = 'Towards East'
                    print(title)
                    print('--------------------')
                    legends.append('East')
                elif compass_val == 3:
                    title = 'Towards South-East'
                    print(title)
                    print('--------------------')
                    legends.append('South-East')
                elif compass_val == 4:
                    title = 'Towards South'
                    print(title)
                    print('--------------------')
                    legends.append('South')
                elif compass_val == 5:
                    title = 'Towards South-West'
                    print(title)
                    print('--------------------')
                    legends.append('South-West')
                elif compass_val == 6:
                    title = 'Towards West'
                    print(title)
                    print('--------------------')
                    legends.append('West')
                elif compass_val == 7:
                    title = 'Towards North-West'
                    print(title)
                    print('--------------------')
                    legends.append('North-West')

            if first_order_learn_mode == 1 or first_order_learn_mode == 3:
                value = compute_gen[0]
            elif first_order_learn_mode == 2:
                value = var_non_scaled[loc[0]] + compute_gen[0]

            if len(var_non_scaled) > 1:
                self_WRF.append(var_non_scaled[loc[0]])
                furthest_WRF.append(var_non_scaled[loc[idx]])
            else:
                self_WRF.append(var_non_scaled[0])
                furthest_WRF.append(var_non_scaled[0])
            furthest_NN.append(value)

            if draw_profiles_diff:
                diff_buff.append(diff)

        for j in range(0, num_vert_layers):
            diff_profiles[nt][j] = diff_buff[j]

        if draw_profiles_diff:
            block = False
        else:
            if nt == do_n_times-1:
                block = True
            else:
                block = False
        line_plot(furthest_WRF, eta, title, block, x2=furthest_NN, y_log=use_log)

    if draw_profiles_diff:
        plt.figure()
        plt.gca().invert_yaxis()
        plt.title('Abs. diff.')
        for i in range(0, do_n_times):
            plt.plot(diff_profiles[i], eta, label=legends[i])

        if use_log:
            plt.yscale('log')
        set_xlabel(True)
        plt.ylabel(r'$\eta$')
        plt.legend(loc='lower right')
        plt.show()


def test_neural_network_first_order(num_vert_profiles=None, time_stamp=None, mode=None):

    num_vert = num_vert_profiles
    if num_test_samples > 1:
        num_vert = 1

    first_time = True
    start = 0
    end = num_vert

    if ignore_timestamp:
        t = ignored_timestamp
    else:
        t = random.randint(0, num_test_time_stamps - 1)
    print('----- Data sampled using time stamp:')
    print(time_stamps[t])

    if num_test_samples > 1:

        NN_inputs, longi, lat = create_test_dataset_first_order(t, num_columns, eta,
                                        non_scaled_layers_files, first_time, num_test_samples=num_test_samples,
                                        lat=Y_non_scaled, long=X_non_scaled)

        nn_gen_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
        print(nn_gen_dataset.tail())

        gen_labels = nn_gen_dataset.pop('TAR')
        # elif order_of_learning == 1:
        #     gen_labels = pd.DataFrame()
        #     loc = 0
        #     for i in range(required_neighbors):
        #         label = 'TAR' + '_' + str(i)
        #         VAR = nn_gen_dataset.pop(label)
        #         gen_labels.insert(loc=loc, column=label, value=VAR)
        #         loc = loc + 1

        print('gen labels:')
        print(gen_labels.tail())

        compute_gen = loaded_model.predict(nn_gen_dataset).flatten()

        error = compute_gen - gen_labels
        # elif order_of_learning == 1:
        #     mean_gen, mean_labels = average_over_outputs(compute_gen, gen_labels)
        #     error = mean_gen - mean_labels

        plt.figure()
        plt.scatter(gen_labels, compute_gen)
        if target == 'U' or target == 'V' or target == 'W':
            plt.xlabel('True values [m/s]')
            plt.ylabel('Predictions [m/s]')
        elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
            plt.xlabel(r'$\mathrm{True\,values}\,[k s^{-1}]$')
            plt.ylabel(r'$\mathrm{Predictions}\,[k s^{-1}]$')
        plt.axis('equal')
        plt.axis('square')
        if target == 'U' or target == 'V' or target == 'W':
            plt.xlim([-50, plt.xlim()[1]])
            plt.ylim([-50, plt.ylim()[1]])
            _ = plt.plot([-100, 100], [-100, 100])
        elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
            plt.xlim([-plt.xlim()[1], plt.xlim()[1]])
            plt.ylim([-plt.ylim()[1], plt.ylim()[1]])
            _ = plt.plot([-10, 10], [-10, 10])
        elif target == 'TURB_MOMENTUM_FLUXES':
            plt.xlim([-plt.xlim()[1], plt.xlim()[1]])
            plt.ylim([-plt.ylim()[1], plt.ylim()[1]])
            _ = plt.plot([-200, 200], [-200, 200])


        plt.show(block=False)

        plt.figure()
        plt.hist(error, bins=25)
        if target == 'U' or target == 'V' or target == 'W':
            plt.xlabel('Prediction error [m/s]')
        elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
            plt.xlabel(r'$\mathrm{Prediction\,error}\,[k s^{-1}]$')

        _ = plt.ylabel("Count")
        plt.show()

    else:
        Data_holder = pd.DataFrame()
        x, y, raw_data = loop_common_1(Data_holder, t, path_to_V, time_stamps, first_time)

        NeighborsTree = loop_common_2(x, y, False)

        non_scaled_at_layer = pd.DataFrame()
        if compute_target:
            compute_turb_moment_fluxes(non_scaled_at_layer, num_vert_layers, path_to_V_non_scaled, time_stamps, t)
        else:
            for l in range(0, num_vert_layers):
                longi, lati = loop_common_3(non_scaled_at_layer, non_scaled_layers_files, False,
                                   t, l)

        if mode == 1:
            first_order_test_mode_1(NeighborsTree, raw_data, Data_holder, non_scaled_at_layer, t)
        elif mode == 2:
            first_order_test_mode_2(NeighborsTree, raw_data, Data_holder, non_scaled_at_layer, t)

# Load json and create model
json_file = open(model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights(model_h5)
print("Loaded model from disk")

loaded_model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse'])

if order_of_learning == 0:
    if compute_draw_2D_map:
        if ignore_timestamp:
            stamp = ignored_timestamp
        else:
            stamp = random.randint(0, num_test_time_stamps-1)
        
        if select_time_stamp >= 0:
            stamp = select_time_stamp

        first_layer = first_2D_map_layer_index
        last_layer = last_2D_map_layer_index + 1
        for l in range(first_layer, last_layer):
            test_neural_network_zero_order(True, map_layer=l, last_layer=last_layer, time_stamp=stamp)
    else:
        test_neural_network_zero_order(False, num_vert_profiles=num_vert_profiles)

elif order_of_learning == 1:
    test_neural_network_first_order(num_vert_profiles=num_vert_profiles, mode=first_order_test_mode)
