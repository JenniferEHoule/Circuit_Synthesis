""" create_netlist.py
This program creates a netlist for a multiport network using the input state space model generated from vectfit3.py

Author: Jennifer Houle
Date: 2/17/2020

[1]  VA. Zadehgol, "A semi-analytic and cellular approach to rational system characterization through
equivalent circuits", Wiley IJNM, 2015. [Online]. https://doi.org/10.1002/jnm.2119

[2]  V. Avula and A. Zadehgol, "A Novel Method for Equivalent Circuit Synthesis from Frequency Response of Multi-port
Networks", EMC EUR, pp. 79-84, 2016. [Online]. Available: ://WOS:000392194100012.

"""

import numpy as np
import scipy

from math import sqrt
from pathlib import Path


def reduce_poles(poles, residues):
    """
    This removes the second half of the complex conjugate pair for the poles and residues.
    Each pair together will make a RLC branch
    :param poles: poles from the vector fitting algorithm
    :param residues: residues from the vector fitting algorithm
    :return: poles, residues with the pairs removed.
    """

    number_of_imaginary_poles = scipy.count_nonzero(poles[0, :].imag)
    for pole in range(0, poles.shape[1] - number_of_imaginary_poles // 2):
        if poles[0, pole].imag:
            poles = np.delete(poles, pole + 1, 1)
            residues = np.delete(residues, pole + 1, 1)
    return poles, residues


def create_real_netlist_branch(pole, residue, port_a, port_b, branch_number, netlist_file):
    """
    Creates a single branch of the netlist from a real pole/residue using Fig. 1 and Eq. (7) in [1].
    :param pole: Real pole
    :param residue: Real residue
    :param port_a: First node
    :param port_b: Second node
    :param branch_number: Reference number based on how many branches have been created so far to uniquely
    identify nodes
    :param netlist_file: The file being written containing the netlist
    :return:
    """
    print(f"* Branch {branch_number}", file=netlist_file)
    print(f"R1br{branch_number} {port_a} "
          f"net{branch_number} {(-pole.real / residue.real)}", file=netlist_file)
    print(f"L1br{branch_number} net{branch_number} "
           f"{port_b} {1 / residue.real}\n", file=netlist_file)


def create_imag_netlist_branch(pole, residue, port_a, port_b, branch_number, netlist_file):
    """
    Creates a single branch of the netlist from an imaginary pole/residue using Fig. 2 and Eq. (18) in [1].
    :param pole: Imaginary pole
    :param residue: Imaginary residue
    :param port_a: First node
    :param port_b: Second node
    :param branch_number: Reference number based on how many branches have been created so far to uniquely
    identify nodes
    :param netlist_file: The file being written containing the netlist
    :return:
    """
    print(f"* Branch {branch_number}", file=netlist_file)
    print(f"Rabr{branch_number} {port_a} "
          f"netRa{branch_number} "
          f"{(residue.imag * pole.imag - residue.real * pole.real) / (2 * (residue.real) ** 2)}", file=netlist_file)
    print(f"Lbr{branch_number} netRa{branch_number} "
          f"netL{branch_number} "
          f"{1 / (2 * residue.real)}", file=netlist_file)
    print(f"Rbbr{branch_number} netL{branch_number} "
          f"{port_b} "
          f"{((pole.imag) ** 2 * ((residue.imag) ** 2 + (residue.real) ** 2)) / (2 * (residue.real) ** 2 * (residue.imag * pole.imag + residue.real * pole.real))}", file=netlist_file)
    print(f"Cbr{branch_number} netL{branch_number} "
          f"{port_b} "
          f"{(2 * (residue.real) ** 3) / ((pole.imag) ** 2 * ((residue.imag) ** 2 + (residue.real) ** 2))}\n", file=netlist_file)


def create_netlist_branch(pole, residue, port_a, port_b, branch_number, netlist_file):
    """
    This chooses which function to call (real or imaginary) to create another branch for the netlist file
    :param pole: Real pole
    :param residue: Real residue
    :param port_a: First node
    :param port_b: Second node
    :param branch_number: Reference number based on how many branches have been created so far to uniquely
    identify nodes
    :param netlist_file: The file being written containing the netlist
    :return:
    """
    if pole.imag:
        create_imag_netlist_branch(pole, residue, port_a, port_b, branch_number, netlist_file)
    else:
        create_real_netlist_branch(pole, residue, port_a, port_b, branch_number, netlist_file)


def create_netlist_file(poles, residues, out_file_path='netlist.sp'):
    """
    This creates a netlist based on poles and residues according to [1].
    :param poles: Array of poles 1 x num_of_poles (entries can be real or imaginary)
    :param residues: Array of residues num_of_y_parameters x num_of_poles (entries can be real or imaginary)
    :return:
    """
    poles, residues = reduce_poles(poles, residues)
    number_of_ports = sqrt(residues.shape[0])
    if residues.shape[0] % number_of_ports != 0:
        raise ValueError("Invalid size of residue array (it must have rows with an integer square root).")
    else:
        number_of_ports = int(number_of_ports)

    outfile = Path(out_file_path)
    with outfile.open('w') as netlist_file:
        print(f"* netlist generated with vector fitting poles and residues\n", file=netlist_file)

        # Main circuit declaration (configuration based on Fig. 2 in [2])
        main_declaration = ".subckt total_network "
        for port_1 in range(1, number_of_ports + 1):
            main_declaration += f"node_{port_1} "
        print(main_declaration + 'node_ref', file=netlist_file)

        for port_1 in range(1, number_of_ports + 1):
            for port_2 in range(1, number_of_ports + 1):
                if port_1 == port_2:
                    port_2_name = 'node_ref'
                else:
                    port_2_name = f'node_{port_2}'
                if port_1 <= port_2:
                    print(f"X_{port_1}{port_2} node_{port_1} {port_2_name} yp{port_1}{port_2}", file=netlist_file)
        print(f".ends\n\n", file=netlist_file)     # End of the main circuit declaration

        # Subcircuits
        for port_1 in range(1, number_of_ports + 1):
            for port_2 in range(1, number_of_ports + 1):
                residue_index = port_1 - 1 + (port_2 - 1) * number_of_ports
                if port_1 == port_2:
                    port_2_name = 'node_ref'
                else:
                    port_2_name = f'node_{port_2}'
                if port_1 <= port_2:
                    print(f"* Y'{port_1}{port_2}", file=netlist_file)
                    print(f".subckt yp{port_1}{port_2} node_{port_1} {port_2_name}", file=netlist_file)
                    for pole_num in range(poles.shape[1]):
                        create_netlist_branch(poles[0, pole_num], residues[residue_index, pole_num], f'node_{port_1}', port_2_name, pole_num, netlist_file)
                    print(f".ends\n\n", file=netlist_file)
        print(f".end", file=netlist_file)

    print(f"Wrote netlist to: {outfile.absolute()}")

# The following was used for testing purposes
# poles = np.load('poles_1port.npy')
# residues = np.load('residues_1port.npy')
# poles = np.load('poles_4port.npy')
# residues = np.load('residues_4port.npy')

# poles = np.array([[1e3, 1e9+1j*1e-7, 1e9-1j*1e-7]])
# residues = np.array([[1, 5 + 1j, 5 - 1j], [3, 7 + 2* 1j, 7 - 1j * 2], [18, 9 + 1j * 3, 9 - 1j * 3], [-3, -7 + 1j, -7 - 1j]])
# create_netlist_file(poles, residues)