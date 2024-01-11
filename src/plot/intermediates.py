from typing import List

import matplotlib.pyplot as plt

def plot_intermediates(state, atom_idx: list):
    """
    plots the intermediate state values for the atoms given by atom_idx
    :param state:
    :param atom_idx:
    :return:
    """
    n_lays = len(state['intermediates'])
    # x-axis
    x = [i for i in range(n_lays+1)]
    # plot intermediates for j-th atom
    for i in atom_idx:
        y = [float(state['intermediates']['layers_0']['x_in'][0][i])]
        for lay_key in state['intermediates'].keys():
            y.append(float(state['intermediates'][lay_key]['x_out'][0][i]))

        plt.plot(x, y)
        plt.figure()
    plt.show()


def plot_intermediates_alt(state, atom_idx: list):
    """
    quick fix for plot_intermediates(state, atom_idx), where stores only one layer in which the intermediate values are listed
    :param state:
    :param atom_idx:
    :return:
    """
    n_lays = len(state['intermediates']['layers_0']['x_in'])
    # x-axis
    x = [i for i in range(n_lays+1)]
    # plot intermediates for j-th atom
    for j in atom_idx:
        y: list[float] = [float(state['intermediates']['layers_0']['x_in'][0][j])]
        for n, lay_key in enumerate(state['intermediates']['layers_0']['x_out']):
            y.append(float(state['intermediates']['layers_0']['x_out'][n][j]))

        plt.plot(x, y)
        plt.figure()
    plt.show()