from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from datetime import datetime

from prep_dataset_train import *

def save_model(path):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # Serialize model to json
    model_json = model.to_json()
    file = "{}/model-{}.{}".format(path, now, "json")
    with open(file, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    file = "{}/model-{}.{}".format(path, now, "h5")
    model.save_weights(file)
    print("Saved model to disk...")


num_training_time_stamps = len(time_stamps)

num_hidden_layers = 2
num_nodes = 32
num_eta_nodes = 16

# The hidden layers after concatenation
unified_num_hidden_layers = 1
unified_num_nodes = 100

eta = get_eta(num_vert_layers, path_to_eta)
print(eta)

non_scaled_layers_files = get_non_scaled_layers_files(num_vert_layers)

if order_of_learning == 0:
    num_examples = num_training_time_stamps * (num_training_samples*time_stamp_sampling_times) * num_vert_layers
elif order_of_learning == 1:
    num_examples = num_training_time_stamps * num_training_samples*number_grid_points_apart * \
                   required_neighbors * num_vert_layers

print('Total number of examples used for training:')
print(num_examples)

if compute_kdTree_from_global_grid:
    print('kd-tree constructed from the global grid')
else:
    print('kd-tree constructed from the sampled data')

if order_of_learning == 0:
    print('learn zero order')
elif order_of_learning == 1:
    print('learn first order')
else:
    print('unsupported learning order')
    sys.exit(-1)


# Compute the number of inputs to the network
num_columns = compute_num_columns()

NN_inputs = [[0 for j in range(num_columns)] for i in range(num_examples)]

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

if order_of_learning == 0:
    create_train_dataset_zero_order(NN_inputs, eta, num_training_time_stamps, time_stamp_sampling_times,
                           num_training_samples, non_scaled_layers_files, num_columns)
elif order_of_learning == 1:
    create_train_dataset_first_order(NN_inputs, eta, num_training_time_stamps, time_stamp_sampling_times,
                           num_training_samples, non_scaled_layers_files, num_columns, lat=Y_non_scaled, long=X_non_scaled)


NN_columns = define_nn_columns()

NN_dataset = pd.DataFrame(NN_inputs, columns=NN_columns)
print(NN_dataset.tail())

train_dataset = NN_dataset.sample(frac=0.8)
print('Train dataset sample:')
print(train_dataset.tail())

test_dataset = NN_dataset.drop(train_dataset.index)
print('Test dataset sample:')
print(test_dataset.tail())

train_labels = train_dataset.pop('TAR')
test_labels = test_dataset.pop('TAR')
# elif order_of_learning == 1:
#     train_labels = pd.DataFrame()
#     test_labels = pd.DataFrame()
#     loc = 0
#     for i in range(required_neighbors):
#         label = 'TAR' + '_' + str(i)
#         VARTRAIN = train_dataset.pop(label)
#         VARTEST = test_dataset.pop(label)
#         train_labels.insert(loc=loc, column=label, value=VARTRAIN)
#         test_labels.insert(loc=loc, column=label, value=VARTEST)
#         loc = loc + 1


print('Train labels:')
print(train_labels.tail())
print('Test labels:')
print(test_labels.tail())

if order_of_learning == 0:
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

    final_nn_inputs(train_dataset, required_neighbors, V1_Train_set, V2_Train_set, V3_Train_set, V4_Train_set,
                V5_Train_set, V6_Train_set, D_Train_set, ETA_Train_set, True, test_dataset,
                V1_Test_set, V2_Test_set, V3_Test_set, V4_Test_set, V5_Test_set, V6_Test_set, D_Test_set, ETA_Test_set)


def add_hidden_layers(prev_layer, n_hidden_layers, n_nodes, activation, regularization):
    x = Dense(num_nodes, activation=activation, kernel_regularizer=regularizers.l2(regularization))(prev_layer)
    for nl in range(n_hidden_layers-1):
        x = Dense(n_nodes, activation=activation, kernel_regularizer=regularizers.l2(regularization))(x)
    return x


def model_zero_order():
    vec_in = []
    vec_channel = []
    var_idx = 0
    for item in V_dict.values():
        if item == 1:
            if (var_idx + 1) == 1:
                v1_input = Input(shape=(required_neighbors,), dtype='float32', name='v1_input')
                v1_channel = add_hidden_layers(v1_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
                vec_in.append(v1_input)
                vec_channel.append(v1_channel)
            if (var_idx + 1) == 2:
                v2_input = Input(shape=(required_neighbors,), dtype='float32', name='v2_input')
                v2_channel = add_hidden_layers(v2_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
                vec_in.append(v2_input)
                vec_channel.append(v2_channel)
            if (var_idx + 1) == 3:
                v3_input = Input(shape=(required_neighbors,), dtype='float32', name='v3_input')
                v3_channel = add_hidden_layers(v3_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
                vec_in.append(v3_input)
                vec_channel.append(v3_channel)
            if (var_idx + 1) == 4:
                v4_input = Input(shape=(required_neighbors,), dtype='float32', name='v4_input')
                v4_channel = add_hidden_layers(v4_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
                vec_in.append(v4_input)
                vec_channel.append(v4_channel)
            if (var_idx + 1) == 5:
                v5_input = Input(shape=(required_neighbors,), dtype='float32', name='v5_input')
                v5_channel = add_hidden_layers(v5_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
                vec_in.append(v5_input)
                vec_channel.append(v5_channel)
            if (var_idx + 1) == 6:
                v6_input = Input(shape=(required_neighbors,), dtype='float32', name='v6_input')
                v6_channel = add_hidden_layers(v6_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
                vec_in.append(v6_input)
                vec_channel.append(v6_channel)
        var_idx = var_idx + 1

    if include_distance:
        dist_input = Input(shape=(required_neighbors,), dtype='float32', name='dist_input')
        dist_channel = add_hidden_layers(dist_input, num_hidden_layers, num_nodes, activation_func, regularization_fac)
        vec_in.append(dist_input)
        vec_channel.append(dist_channel)

    eta_input = Input(shape=(1,), dtype='float32', name='eta_input')
    eta_channel = add_hidden_layers(eta_input, num_hidden_layers, num_eta_nodes, activation_func, regularization_fac)
    vec_in.append(eta_input)
    vec_channel.append(eta_channel)

    concat_layer = keras.layers.concatenate(vec_channel)

    x = add_hidden_layers(concat_layer, unified_num_hidden_layers, unified_num_nodes, activation_func, regularization_fac)

    # The output
    main_output = Dense(1, name='main_output')(x)

    model = Model(inputs=vec_in, outputs=main_output)
    return model


def model_first_order():
    inputs = Input(shape=(num_columns-1,), dtype='float32', name='inputs')
    hidden = add_hidden_layers(inputs, num_hidden_layers, num_nodes, 'relu', regularization_fac)
    # The output
    main_output = Dense(1, name='main_output')(hidden)

    model = Model(inputs=inputs, outputs=main_output)
    return model


def build_model():
    if order_of_learning == 0:
        model = model_zero_order()
    elif order_of_learning == 1:
        model = model_first_order()

    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mae', 'mse'])
    return model


model = build_model()
print(model.summary())

# Try out a dummy model
if order_of_learning == 0:
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
elif order_of_learning == 1:
    example_batch_input = train_dataset[:10]
    example_result = model.predict(example_batch_input)

print(example_result)

# Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end='')


if order_of_learning == 0:
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
elif order_of_learning == 1:
    history = model.fit(
        x=train_dataset,
        y=train_labels,
        epochs=EPOCHS, batch_size=batch_size, validation_split=0.2,
        verbose=0, callbacks=[PrintDot()]
        )

print('\n')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# Evaluate using test set
if order_of_learning == 0:
    loss, mae, mse = model.evaluate(
                    x=vec_test,
                    y=test_labels, verbose=0
                )
elif order_of_learning == 1:
    loss, mae, mse = model.evaluate(
                    x=test_dataset,
                    y=test_labels, verbose=0
                )

print("Testing set Mean Abs Error: {:5.2f} m/s", format(mae))

# Save the model to disk
save_model(path_to_saved_model)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    if target == 'U' or target == 'V' or target == 'W': 
        plt.ylabel('Mean Abs Error [m/s]')
    elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
        plt.ylabel(r'$\mathrm{Mean\,Abs\,Error}\,[k s^{-1}]$')
    elif target == 'TURB_MOMENTUM_FLUXES':
        plt.ylabel(r'$\mathrm{Mean\,Abs\,Error}\,[m^{2} s^{-2}]$')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.legend()
    if first_order_learn_mode == 2:
        plt.ylim([0, 1])
    else:
        if target == 'U' or target == 'V' or target == 'W':
            plt.ylim([0, 20])
        elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VOL':
            plt.ylim([0, 0.1])
    plt.show(block=False)

    plt.figure()
    plt.xlabel('Epoch')
    if target == 'U' or target == 'V' or target == 'W':
        plt.ylabel('Mean Square Error [$(m/s)^2$]')
    elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
        plt.ylabel(r'$\mathrm{Mean\,Abs\,Error}\,[(k s^{-1})^{2}]$')
    elif target == 'TURB_MOMENTUM_FLUXES':
        plt.ylabel(r'$\mathrm{Mean\,Abs\,Error}\,[(m^{2} s^{-2})^{2}]$')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.legend()
    if first_order_learn_mode == 2:
        plt.ylim([0, 5])
    else:
        if target == 'U' or target == 'V' or target == 'W':
            plt.ylim([0, 100])
        elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
            plt.ylim([0, 0.1])
    plt.show(block=False)


plot_history(history)

if order_of_learning == 0:
    test_predictions = model.predict(vec_test).flatten()
elif order_of_learning == 1:
    test_predictions = model.predict(test_dataset).flatten()


plt.figure()
plt.scatter(test_labels, test_predictions)
error_test = test_predictions - test_labels
# elif order_of_learning == 1:
#     mean_predict, mean_labels = average_over_outputs(test_predictions, test_labels)
#     plt.scatter(mean_labels, mean_predict)
#     error_test = mean_predict - mean_labels


if target == 'U' or 'target' == 'W' or target == 'W':
    plt.xlabel('True values [m/s]')
    plt.ylabel('Predictions [m/s]')
elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
    plt.xlabel(r'$\mathrm{True\,values}\,[k s^{-1}]$')
    plt.ylabel(r'$\mathrm{Predictions}\,[k s^{-1}]$')
plt.axis('equal')
plt.axis('square')
if target == 'U' or 'target' == 'W' or target == 'W':
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
plt.hist(error_test, bins=25)
if target == 'U' or 'target' == 'W' or target == 'W':
    plt.xlabel('Prediction error [m/s]')
elif target == 'REL_VERT_VORT' or target == 'ABS_VERT_VORT':
    plt.xlabel(r'$\mathrm{Prediction\,error}\,[k s^{-1}]$')
_ = plt.ylabel("Count")
plt.show()



