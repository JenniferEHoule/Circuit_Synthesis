""" 4port_example_ADS.py
4 port data taken from ADS MATLAB output

This takes 4 port ADS MATLAB data
1. Extracts the frequency and Y data
2. Converts it to a Y' matrix (and converts frequency to s)
3. Runs the Vector Fitting Algorithm to generate poles and residues
4. Uses the poles and residues to generate a netlist

Optional - part of the code may be added to read in the new data from ADS (generated with the netlist) and compare
to the original data

Uses the Vector Fitting algorithm and is based partially off the example ex4a.m
(details below author information)
This implements the VFIT 3 algorithm for a 4-port system.

Date: 3/7/2020
Author: Jennifer Houle

B. Gustavsen, VFIT3, The Vector Fitting Website. March 20, 2013. Accessed on:
Jan. 21, 2020. [Online]. Available: https://www.sintef.no/projectweb/vectfit/downloads/vfut3/.

This example script is part of the vector fitting package (VFIT3.zip)
Last revised: 08.08.2008.
Created by:   Bjorn Gustavsen.

"""


import numpy as np

from vectfit3 import VectorFit3
from create_netlist import create_netlist_file
from convert_y_to_y_prime import convert_y_to_y_prime_matrix_values
from plots import plot_magnitude_phase_VF_format

s = np.load('output_from_4port_simulation_original_s.npy')
f = np.load('output_from_4port_simulation_original_f.npy')
s_original = s.copy()
f_original = f.copy()

f = convert_y_to_y_prime_matrix_values(f)

s = 1j * s * 2 * np.pi

# Input data
Nc = f.shape[0]
Ns = s.shape[0]

# Start of Vector Fitting Part of Program

N = 10 # Order of approximation
# Complex starting poles
w = s / 1j
bet = np.linspace(w[0], w[Ns-1], N//2)
poles = np.zeros(N, dtype=np.complex)
for n in range(N//2):
    alf = -bet[n] * 1e-2
    poles[2 * n] = alf - bet[n] * 1j
    poles[2 * n + 1] = alf + bet[n] * 1j

poles = poles.reshape((1, -1))

# Parameters for Vector Fitting
weight = 1/np.sqrt(np.abs(f))

Niter=5
rms = np.zeros((Niter, 1))
for iter in range(Niter):
    print(f'Iteration: {iter}')
    vector_fitter = VectorFit3(asymp='NONE',
                               phaseplot=(iter == Niter - 1),
                               errplot=False,
                               complex_ss=False,
                               logx=True, logy=True,
                               skipres=(iter != Niter - 1),
                               spy2=(iter == Niter - 1))
    SER, poles, rmserr, fit = vector_fitter.do_fit(f, s, poles, weight)
    rms[iter,0] = rmserr

create_netlist_file(poles, SER['C'])

# Use the following code if running a final comparison between the original simulation output and
# the netlist simulation output.
# Note: final_s and final_f were taken from an ADS simulation running the netlist
# generated with create_netlist_file. This is only an example of how one would make the comparison
# final_s = np.load('output_from_4port_simulation_netlist_s.npy')
# final_f = np.load('output_from_4port_simulation_netlist_f.npy')
#
# plot_magnitude_phase_VF_format(s_original, f_original, final_f, final_f.shape[0], Ns, "Original", "Final", "example_4_port_ADS")
#
