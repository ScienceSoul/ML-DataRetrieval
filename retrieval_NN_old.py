from __future__ import absolute_import, division, print_function

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load(path1, path2):
    csv_path = os.path.join(path1, path2)
    print(csv_path)
    return pd.read_csv(csv_path)

UNITS = ['K']

x_coord = 'longitude'
y_coord = 'latitude'

path_to_U = '/home/seddik/Documents/workdir/WRF_Jebi/NN/2018-09-03_23_00_00/U'
path_to_V = '/home/seddik/Documents/workdir/WRF_Jebi/NN/2018-09-03_23_00_00/V'
path_to_eta = '/home/seddik/Documents/workdir/WRF_Jebi/NN/2018-09-03_23_00_00/ZNU'
path_to_T = '/home/seddik/Documents/workdir/WRF_Jebi/VISUAL/2018-09-03_23_00_00/PERT_T'

num_layers = 50
eta = []
for i in range(num_layers):
    file = str(i) + '.csv'
    data = load(path_to_eta, file)
    z = data.ZNU
    eta.append(z[0])

eta.reverse()
print(eta)

raw_U = load(path_to_U, '0.csv')
X = raw_U[x_coord]
Y = raw_U[y_coord]
U = raw_U.U

raw_V = load(path_to_V, '0.csv')
V = raw_V.V

x_col = []
y_col = []
z_col = []
u_col = []
v_col = []
t_col = []
num_rows = len(X)*num_layers

for i in range(3):
    file = str(i) + '.csv'
    data = load(path_to_T, file)
    T = data.PERT_T
    for j in range(len(U)):
        x_col.append(X[j])
        y_col.append(Y[j])
        z_col.append((eta[i]))
        u_col.append(U[j])
        v_col.append(V[j])
        t_col.append(T[j])

data_tuples =list(zip(x_col,y_col,z_col,u_col,v_col,t_col))
dataset = pd.DataFrame(data_tuples, columns=['x','y','z','u','v','t'])
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
print(train_dataset.tail())
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
print(train_stats.pop('t'))
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('t')
test_labels = test_dataset.pop('t')

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
    return model

model = build_model()
print(model.summary())

# Try out the model
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
print(example_result)

# Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 2000

history = model.fit(
          train_dataset, train_labels,
          epochs=EPOCHS, batch_size=50, validation_split=0.2, verbose=0,
          callbacks=[PrintDot()]
         )

print('\n')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# Evaluate using test set
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} Kelvin", format(mae))

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [K]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label="Train Error")
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.legend()
    plt.ylim([0,2])
    plt.show(block=False)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$k^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.legend()
    plt.ylim([0,8])
    plt.show(block=False)

plot_history(history)

test_predictions = model.predict(test_dataset).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [K]')
plt.ylabel('Predictions [k]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-20,20])
plt.ylim([-20,20])
_ = plt.plot([-100, 100], [-100, 100])
plt.show(block=False)

error_test = test_predictions - test_labels
plt.figure()
plt.hist(error_test, bins=25)
plt.xlabel('Prediction Error [k]')
_ = plt.ylabel("Count")
plt.show(block=False)

# Generate potential temperature at a given layer with the neural network
gen_data = load(path_to_T, '2.csv')
gen_T = gen_data.PERT_T

gen_x_col = []
gen_y_col = []
gen_z_col = []
gen_u_col = []
gen_v_col = []
gen_t_col = []
for i in range(len(U)):
    gen_x_col.append(X[i])
    gen_y_col.append(Y[i])
    gen_z_col.append((eta[2]))
    gen_u_col.append(U[i])
    gen_v_col.append(V[i])
    gen_t_col.append(gen_T[i])

gen_data_tuples =list(zip(gen_x_col,gen_y_col,gen_z_col,gen_u_col,gen_v_col,gen_t_col))
gen_dataset = pd.DataFrame(gen_data_tuples, columns=['x','y','z','u','v','t'])

gen_labels = gen_dataset.pop('t')

gen_field = model.predict(gen_dataset).flatten()
plt.figure()
plt.scatter(gen_labels, gen_field)
plt.xlabel('True Values [K]')
plt.ylabel('Predictions [K]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-20,20])
plt.ylim([-20,20])
_ = plt.plot([-100, 100], [-100, 100])
plt.show(block=False)

error_gen = gen_field - gen_labels
plt.figure()
plt.hist(error_gen, bins=25)
plt.xlabel('Prediction Error [K]')
_ = plt.ylabel("Count")
plt.show(block=False)

# Plot the generated field
xlabel = 'Longitude (degrees east)'
ylabel = 'Latitude (degrees north)'
fig = plt.figure(figsize=(20,15))
var_plt = plt.scatter(X,Y,c=gen_field, cmap='seismic')
var_plt.set_clim(np.min(gen_labels), np.max(gen_labels))
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
cbar = plt.colorbar(var_plt)
cbar.ax.tick_params(labelsize=12)

bar_title = 'T' + ' ' + '(' + UNITS[0] + ')'
cbar.ax.set_title(bar_title)
plt.show()

