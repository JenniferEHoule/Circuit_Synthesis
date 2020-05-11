""" convert_y_to_y_prime.py
This file will convert a Y matrix to a Y' matrix. This is based on Eq. (9) of [2].

Date: 2/28/2020
Author: Jennifer Houle

[2]  V. Avula and A. Zadehgol, "A Novel Method for Equivalent Circuit Synthesis from Frequency Response of Multi-port
Networks", EMC EUR, pp. 79-84, 2016. [Online]. Available: ://WOS:000392194100012.
"""

from math import sqrt

import numpy as np


def convert_y_to_y_prime_matrix_values(y_matrix):
    """
    This converts the Y parameters to Y' parameters as in Eq. (9) of [2]
    :param y_matrix: Matrix of Y parameters (num_of_freq_points x num_Y_parameters)
    :return: Matrix of Y' parameters (num_of_freq_points x num_Y_parameters)
    """
    number_of_ports = sqrt(y_matrix.shape[0])
    if y_matrix.shape[0] % number_of_ports != 0:
        raise ValueError("Invalid size of residue array (it must have rows with an integer square root).")
    else:
        number_of_ports = int(number_of_ports)
    y_prime_matrix = np.zeros((y_matrix.shape[0], y_matrix.shape[1]), dtype=complex)
    for port_1 in range(1, number_of_ports + 1):
        for port_2 in range(1, number_of_ports + 1):
            y_index = port_1 - 1 + (port_2 - 1) * number_of_ports
            if port_1 == port_2:
                y_prime_matrix[y_index, :] = \
                    np.sum(y_matrix[(port_1 - 1) * number_of_ports:port_1 * number_of_ports, :], axis=0)
            else:
                y_prime_matrix[y_index, :] = -y_matrix[y_index, :].copy()
    return y_prime_matrix


def convert_y_to_y_prime_matrix_values_3d(y_matrix):
    """
    This converts the Y parameters to Y' parameters as in Eq. (9) of [2]
    :param y_matrix: Matrix of Y parameters (num_ports x num_ports x num_of_freq_points)
    :return: Matrix of Y' parameters (num_ports x num_ports x num_of_freq_points)
    """
    number_of_ports = y_matrix.shape[0]
    if y_matrix.shape[0] != y_matrix.shape[1]:
        raise ValueError("Invalid size of residue array (it must have the same first and second dimension).")
        return

    y_prime_matrix = np.zeros((number_of_ports, number_of_ports, y_matrix.shape[2]), dtype=complex)
    for port_1 in range(0, number_of_ports):
        for port_2 in range(0, number_of_ports):
            if port_1 == port_2:
                y_prime_matrix[port_1, port_2, :] = \
                    np.sum(y_matrix[port_1, :, :], axis=0)
            else:
                y_prime_matrix[port_1, port_2, :] = -y_matrix[port_1, port_2, :].copy()
    return y_prime_matrix
