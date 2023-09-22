import sys
from global_defs import *
from JEBI_params import *

import matplotlib.pyplot as plt

def plot_vertical_profile(var_name, time_stamps, t, file_suffix, vert_coord, units, 
                          nx, i, j, block, y_log=None):
    
    path = path_to_V_non_scaled + time_stamps[t] + '/' + var_name

    data = []
    for l in range(num_vert_layers):
        file = str(l) + file_suffix
        raw = load(path, file, False)
        buff = raw.pop(var_name)
        data.append(buff[(nx*j)+i])
    
    plt.figure()
    plt.gca().invert_yaxis()
    plt.title('Vertical profile')
    plt.plot(data, vert_coord, 'b', label=var_name)
    if y_log:
        if y_log == True:
            plt.yscale('log')

    xlabel = var_name + ' ' + units
    plt.xlabel(xlabel)
    plt.ylabel(r'$\eta$')
    plt.legend(loc='lower left')
    if block == False:
            plt.show(block=False)
    else:
        plt.show()

def display_profile_abs_velocity(time_stanps, t, file_suffix, vert_coord, units, nx, i, j, block,
    y_log=None):
    
    path_u = path_to_V_non_scaled + time_stamps[t] + '/' + 'U'
    path_v = path_to_V_non_scaled + time_stamps[t] + '/' + 'V'

    abs_velo = []
    for l in range(num_vert_layers):
        file = str(l) + file_suffix
        raw_u = load(path_u, file, False)
        raw_v = load(path_v, file, False)
        buff_u = raw_u.pop('U')
        buff_v = raw_v.pop('V')
        u = buff_u[(nx*j)+i]
        v = buff_v[(nx*j)+i]
        velo = sqrt(u**2 + v**2)
        abs_velo.append(velo)

    plt.figure()
    plt.gca().invert_yaxis()
    plt.title('Vertical profile')
    plt.plot(abs_velo, vert_coord, 'b', label='Abs. Velo.')
    if y_log:
        if y_log == True:
            plt.yscale('log')

    xlabel = 'Abs. Velo.' + ' ' + units
    plt.xlabel(xlabel)
    plt.ylabel(r'$\eta$')
    plt.legend(loc='lower left')
    if block == False:
            plt.show(block=False)
    else:
        plt.show()
