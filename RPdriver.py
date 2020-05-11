""" RPdriver.py

Author: Jennifer Houle
Date: 3/19/2020

This program is based off RPdriver.m from [4]. From [4],

PURPOSE : Perturb eigenvalues of R, D, and E of an Nth order pole-residue model with Nc ports

                 N   Rm
         Y(s)=SUM(---------  ) +D +s*E
                m=1 (s-am)

          - to enforce passivity: eig(real(Y))>0 for all frequencies
          - to enforce a positive definite D (asymptotic passivity)
          - to enforce a positive definite E

This ensures that the model will not cause instabilities when used in a time domain
simulation.


[1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
    domain responses by Vector Fitting", IEEE Trans. Power Delivery,
    vol. 14, no. 3, pp. 1052-1061, July 1999.

[2] B. Gustavsen, "Improving the pole relocating properties of vector
    fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
    July 2006.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
    "Macromodeling of Multiport Systems Using a Fast Implementation of
    the Vector Fitting Method", IEEE Microwave and Wireless Components
    Letters, vol. 18, no. 6, pp. 383-385, June 2008.

[4] B. Gustavsen, Matrix Fitting Toolbox, The Vector Fitting Website.
    March 20, 2013. Accessed on: Feb. 25, 2020. [Online]. Available:
    https://www.sintef.no/projectweb/vectorfitting/downloads/matrix-fitting-toolbox/.

[5] B. Gustavsen, "Fast passivity enforcement for S-parameter models by perturbation
    of residue matrix eigenvalues",
    IEEE Trans. Advanced Packaging, accepted for publication.

[6] B. Gustavsen, "Fast Passivity Enforcement for Pole-Residue Models by Perturbation
    of Residue Matrix Eigenvalues", IEEE Trans. Power Delivery, vol. 23, no. 4,
    pp. 2278-2285, Oct. 2008.

"""
from enum import Enum, auto
from math import pi

import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

from pr2ss import pr2ss
from rot import rot
from intercheig import intercheig
from pass_check import pass_check_Y, separate_real_imag_in_state_space
from violextrema import violextremaY
from FRPY import FRPY
from utils import WeightParam, OutputLevel


class ParameterType(Enum):
    """
    This class contains the options for parameter type.
    'y' : Y-parameter model
    's' : S-parameter model
    """
    y = 'y'
    s = 's'

class Method(Enum):
    """
    This class contains the options for the method.
    'FRP' : Fast Residue Perturbation method
    'FMP' : I don't support this; I did not see these files in the MATLAB code
    """
    FRP = 'FRP'
    FMP = 'FMP'

class RPdriver:
    """
    This implements the Vector Fitting algorithm
    """
    DEFAULT_OPTIONS = dict(
        parametertype=ParameterType.y,  # Y-parameter is the default (and only version currently supported)
        Niter_out=10,                   # Outer loop iterations (maximum) that generates a list of frequencies where passivity is enforced [6]
        Niter_in=0,                     # Inner loop iterations (maximum) that gererates a list of frequencies where new, negative eigenvalue minima appear [6]
        TOLGD=1e-6,                     # The amount by which eigenvalues of G(s) and D that are negative are shifted positive
        TOLE=1e-12,                     # The amount by which eigenvalues of E that are negative are shifted positive
        complx_ss=True,                 # Use a complex state space model. Note- only this option is currently supported
        weightfactor=0.001,             # Weight for out-of-band auxiliary samples
        weightparam=WeightParam.common_1, # Type of weight array used
        method=Method.FRP,              # Only FRP is supported; I couldn't find the FMP code
        colinterch=True,                # True is the only valid option
        outputlevel=OutputLevel.max,    # Display the amount of text while the program runs
        plot=True,                      # This turns on (True) or off (False) plots
        s_pass=None,                    # Array of frequency samples (optional)
        ylim=None,                      # The eigenvalue plots are limited to this band if used
        xlim=None,                      # The eigenvalue plots are limited to this range if used
        weight=None                     # Weighting array (optional)
    )

    def __init__(self, **options):
        """
        Sets up the options by merging the default options with any the user selects.
        User selected options have priority
        """
        if 'parametertype' in options:
            options['parametertype'] = ParameterType(options['parametertype'])
        if 'weightparam' in options:
            options['weightparam'] = WeightParam(options['weightparam'])
        if 'outputlevel' in options:
            options['outputlevel'] = OutputLevel(options['outputlevel'])
        if 'method' in options:
            options['method'] = Method(options['method'])
        self.options = {**self.DEFAULT_OPTIONS, **options}

        if self.options['method'] == Method.FMP and options['parametertype'] == ParameterType.s:
            print("Error in RPdriver.py: FMP cannot be used together with S-paramters. Program must stop!")

    def rpdriver(self, SER, s):
        """
        This is where the FRPY will be implemented and checks for passivity conducted.
        :param SER: State space model (usually an input from running VFdriver.py)
        :param s: Frequencies across which the state space model was generated.
        :return:
            SER1: The new state space model that is now passive (assuming enough iterations were run; verify with text output)
            bigYfit: The fit with the newly generated state space model
            self.options: Saves off the options used in generating the state space model
        """
        SER = pr2ss(SER)
        print("Starting!")
        colinterch = self.options['colinterch']
        if self.options['parametertype'] == ParameterType.y:
            print("Y parameters")
        elif self.options['parametertype'] == ParameterType.s:
            print("S parameters; this is currently not supported!")
            return

        plotte = False
        if self.options['plot']:
            plotte = True
            s_pass = self.options['s_pass']
            xlimflag = self.options['xlim']
            ylimflag = self.options['ylim']

        SER0 = SER
        Nc = SER['D'].shape[0]

        Niter_out = self.options['Niter_out']
        Niter_in = self.options['Niter_in']

        # Plotting eigenvalues of original model (SERC0, SERD0):
        if plotte == True:
            oldT0 = None
            oldU = None
            I = np.ones((SER['A'].shape[0], 1))
            EE0 = np.zeros((Nc, s_pass.shape[0]), dtype=complex)
            for k in range(s_pass.shape[0]):
                Y = self.calculate_y_from_SER(I, SER, k, s_pass) # Implements Eq. (15) in [6]
                if self.options['parametertype'] == ParameterType.y:
                    # Fills in the eigenvalues into EE0
                    EE0, oldT0 = self.calculate_eigenvalues_for_EE(EE0, Nc, Y, k, oldT0)
                elif self.options['parametertype'] == ParameterType.s:
                    print("S Type parameters not currently supported!")
                    return
                    if self.options['colinterch']:
                        EE0[:, k] = LA.svd(Y, 0)
                    else:
                        U, S, V = LA.svd(Y, 0)
                        U, S, V = intercheig(U, OldU, S, V, Nc, k)
                        oldU = U
                        EE0[:, k] = np.diagflat(S)

            self.plot_eigenvalues_of_gs(s_pass, EE0, xlimflag, ylimflag)

        # Passivity Enforcement
        SER1 = SER0
        break_outer = False

        # Outer loop: generates a list of frequencies where passivity is enforced [6]
        for iter_out in range(1, Niter_out + 1):
            if break_outer == True:
                # If no adjustments were made, break out of the loop
                SER0 = SER1
                break

            s3 = np.array([])
            # Inner Loop: gererates a list of frequencies where new, negative eigenvalue minima appear [6]
            for iter_in in range(1, Niter_in + 2):
                s2 = np.array([])
                if self.options['outputlevel'] == OutputLevel.max:
                    print(f"[Iterations (Out): {iter_out}, (In): {iter_in - 1}]\n  Passivity Assignment:")

                # For the first iteration,
                if iter_in == 1:
                    if self.options['parametertype'] == ParameterType.y:
                        # Find the violating frequency intervals. Each interval is a column in the matrix wintervals
                        wintervals = pass_check_Y(SER['poles'], SER1['R'], SER1['D'])
                    else:
                        TOL = 1e-3
                        spy = 1
                        print("Parameter Type: S not currently supported!")
                    if np.any(wintervals):
                        if self.options['outputlevel'] == OutputLevel.max:
                            print(f"N.o. violating intervals: {wintervals.shape[1]}")

                    if self.options['parametertype'] == ParameterType.y:
                        eigenvalsD, eigenvectors = LA.eig(SER1['D'])
                        eigenvalsE, eigenvectors = LA.eig(SER1['E'])
                        if not np.any(wintervals) and np.all(eigenvalsD >= 0) and np.all(eigenvalsE >= 0):
                            # If there are no violations, break out of the loop here. SER0 = SER1 indicates no violations
                            SER0 = SER1
                            break_outer = True
                            break
                        elif self. options['parametertype'] == ParameterType.s:
                            print("Option S Parameter type not currently supported!")

                    if self.options['parametertype'] == ParameterType.y:
                        # Identifying minima within each interval:
                        s_viol, g_pass, ss = violextremaY(wintervals.T, SER['poles'], SER1['R'], SER1['D'], colinterch)
                        s2 = s_viol.T.copy()
                        s2 = np.sort(s2) # s2 is the frequency at which the minimum occurs
                        if s2.shape[0] == 0 and np.all(LA.eig(SER1['D']) > 0):
                            break
                    elif self.options['parametertype'] == ParameterType.s:
                        print("Option S Parameter type not currenlty supported!")

                    if self.options['outputlevel'] == OutputLevel.max:
                        self.print_max_violation_for_SER_eig(SER0, SER1, g_pass, ss)
                    if self.options['outputlevel'] != OutputLevel.max:
                        self.print_eigenvalue_violations(SER0, SER1, g_pass)

                if self.options['outputlevel'] == OutputLevel.max:
                    print(" Passivity Enforcement ...")
                if self.options['method'] == Method.FMP:
                    print(" FMP driver not currently supported!")
                    # SER1, MPopts = FMP(SER0, s, s2, s3, MPopts)
                elif self.options['method'] == Method.FRP:
                    if self.options['parametertype'] == ParameterType.y:
                        # This is where the FRP method is used and perturbations occur
                        SER1, MPopts = FRPY(SER0, s, s2, s3, self.options)
                    else:
                        print("Option S Parameter type not currently supported!")
                else:
                    print("*** Error! RMP and FRP are the only valid method options!")

                EE1 = np.zeros((SER1['C'].shape[0], s_pass.shape[0]), dtype=complex)

                if plotte == True:
                    # Plot the new eigenvalues
                    if self.options['parametertype'] == ParameterType.y:
                        oldT0 = np.array([])
                        tell = 0
                        I = np.ones((SER['A'].shape[0], 1))
                        EE1, oldT0 = self.calculate_ee1_for_y_parameter(I, Nc, SER1, oldT0, s_pass, EE1)
                    elif self.options['parametertype'] == ParameterType.s:
                        oldU = np.array([])
                        I = np.ones((SER['A'].shape[0], 1))
                        EE1, oldU = self.calculate_ee1_for_s_parameter(EE0, EE1, I, Nc, SER1, colinterch, k, s, s_pass, oldU)

                    self.plot_figure_8(s_pass, EE0, EE1, xlimflag, ylimflag)

                if iter_in != Niter_in + 1: # Not last run in inner-loop
                    # Find the violating intervals, the minimum, and the frequency at which the minimum occurs
                    # This is for the perturbed model
                    if self.options['parametertype'] == ParameterType.y:
                        wintervals = pass_check_Y(SER1['poles'], SER1['R'], SER1['D'])
                        s_viol, g_pass, ss = violextremaY(wintervals.T, SER1['poles'], SER1['R'], SER1['D'], colinterch)

                    elif self.options['parametertype'] == ParameterType.s:
                        print("Parameter Type S not currently supported")
                    s3 = np.append(s3, s2)
                    s3 = np.append(s3, s_viol.T)

                if iter_in == Niter_in + 1:
                    # If this is the last iteration (inner loop)
                    s3 = np.array([])
                    if plotte == True:
                        EE0 = EE1.copy() # Update model
                    SER0 = SER1.copy()

        # Plotting eigenvalues of modified model (SERC1, SERD1):
        if plotte == True:
            EE1 = np.zeros((SER1['C'].shape[0], s_pass.shape[0]), dtype=complex)
            if self.options['parametertype'] == ParameterType.y:
                oldT0 = np.array([])
                EE1, oldT0 = self.calculate_ee1_for_y_parameter(I, Nc, SER1, oldT0, s_pass, EE1)
            elif self.options['parametertype'] == ParameterType.s:
                EE1, oldU = self.calculate_ee1_for_s_parameter(EE0, EE1, I, Nc, SER1, colinterch, k, s, s_pass, oldU)

        self.plot_figure_7(s_pass, EE1, xlimflag, ylimflag)

        if not np.any(wintervals):
            # This means there were no violation frequency intervals returned as wintervals
            print("Passivity was successfully enforced.")
            if self.options['outputlevel'] == OutputLevel.max:
                if self.options['parametertype'] == ParameterType.y:
                    print("  Max. violation, eig(G) : None")
                    print("  Max. violation, eig(D) : None")
                    print("  Max. violation, eig(E) : None")
                elif self.options['parametertype'] == ParameterType.s:
                    print("  Max. violation, eig(S) : None")
                    print("  Max. violation, eig(D) : None")
        else:
            # Passivity was not successfully enforced
            print(f"  ***Max. violation, eig(G) : {np.amin(g_pass)}")
            print(f"  ***Max. violation, eig(D) : {np.amin(LA.eig(SER0['D'])[0])}")
            print(f"  ***Max. violation, eig(E) : {np.amin(LA.eig(SER0['E'])[0])}")
            print("--> Iterations terminated before completing passivity enforcement.")
            print("    Increase parameter option Niter_out")

        # Producing Plot
        Ns = s.shape[0]
        bigYfit = np.zeros((Nc, Nc, Ns), dtype=complex)
        I = np.ones((SER['A'].shape[0], 1))
        for k in range(Ns):
            Y = self.calculate_y_from_SER(I, SER1, k, s)  # Implements Eq. (15) in [6]
            bigYfit[:, :, k] = Y.copy()

        # Converting to real-only state-space, if requested
        if self.options['complx_ss'] == False:
            cindex, SER1['A'], SER1['B'], SER1['C'] = separate_real_imag_in_state_space(SER1['A'], SER1['B'], SER1['C'])


        print("------------END------------")
        return SER1, bigYfit, self.options

    def print_eigenvalue_violations(self, SER0, SER1, g_pass):
        """
        Print information about the passivity violations
        :param SER0: Original model
        :param SER1: Perturbed model
        :param g_pass: Frequency range of violations
        :return:
        """
        if self.options['parametertype'] == ParameterType.Y:
            min1 = np.amin(g_pass)
        min2 = np.amin(LA.eigvals(SER1['D']))
        print(f" Max. violation : {np.amin(np.hstack((min1, min2)))}")
        if np.amin(LA.eig(SER0['E'])) < 0:
            print(f" Max. violation, E: {np.amin(LA.eigvals(SER1['E']))}")
        elif self.options['parametertype'] == ParameterType.s:
            print("Option S Parameter type not currently supported!")

    def print_max_violation_for_SER_eig(self, SER0, SER1, g_pass, ss):
        """
        :param SER0: Original model
        :param SER1: Perturbed model
        :param g_pass: Frequency range of violations
        :param ss: Frequency minimum for violation range
        :return:
        """
        if self.options['parametertype'] == ParameterType.y:
            if np.amin(g_pass) < 0:
                print(f" Max. violation, eig(G) : {g_pass} @ {np.round((ss / (2 * pi)).imag, 0)} Hz")
            else:
                print(" Max violation, eig(G) : None")
            if np.amin(LA.eigvals(SER0['D'])) < 0:
                print(f" Max. violation, eig(D) : {np.amin(LA.eigvals(SER1['D']))}")
            else:
                print(" Max violation, eig(D) : None")
            if np.amin(LA.eigvals(SER0['E'])) < 0:
                print(f" Max. violation, eig(E) : {np.amin(LA.eigvals(SER1['E']))}")
            else:
                print(" Max violation, eig(E) : None")
        elif self.options['parametertype'] == ParameterType.s:
            print("Option S Parameter type not currently supported!")

    def calculate_eigenvalues_for_EE(self, EE0, Nc, Y, k, oldT0):
        """
        Calculate the eiginevalues of G(s) and fill into EE0 matrix for the given value of k
        :param EE0: Matrix of the eigenvalues of G(s)
        :param Nc: Number of ports on which data is being fit
        :param Y: The result of Eq. (15) in [6]
        :param k: Index for the iteration number
        :param oldT0: Eigenvectors from pervious iteration
        :return: EE0 with additional data for given k; oldT0 with eigenvectors from the current index k
        """
        G = Y.real
        D, T0 = LA.eig(G) # Calculate eigenvalues / eigenvectors to evaluate Eq. (3) in [6]
        T0 = rot(T0.astype(complex))  # Minimizing phase angle of eigenvectors in least squares sense
        T0, D = intercheig(T0, oldT0, np.diag(D).copy(), Nc, k) # Rearrange the eigenvalues / vectors to smooth them out over frequency
        oldT0 = T0
        EE0[:, k] = np.diag(D)
        return EE0, oldT0

    def calculate_ee1_for_s_parameter(self, EE0, EE1, I, Nc, SER1, colinterch, k, s, s_pass, OldU):
        """ This would be used for S parameters but this wasn't fully implemented yet"""
        for k in range(s_pass.shape[0]):
            Y = self.calculate_y_from_SER(I, SER1, k, s_pass)  # Implements Eq. (15) in [6]
            if colinterch == False:
                U, S, V = LA.svd(Y, 0)
                EE0[:, k] = np.diagflat(S)
            else:
                U, S, V = LA.svd(Y, 0)
                print("interchsvd is not written yet. Answer is incorrect.")
                # U, S, V = interchsvd(U, OldU, S, V, Nc, k) # This isn't written yet
                oldU = U
                EE1[:, k] = S
        return EE1, oldU

    def calculate_y_from_SER(self, I, SER1, k, s_pass):
        """ Implements Eq. (15) in [6] """
        Y = SER1['C'] @ np.diagflat((s_pass[k] * I - np.diag(SER1['A']).reshape(-1, 1)) ** (-1)) @ SER1['B'] + SER1['D'] + s_pass[k] * SER1['E']
        return Y

    def calculate_ee1_for_y_parameter(self, I, Nc, SER1, oldT0, s_pass, EE1):
        """
        Calculates the eigenvalue matrix for the Y-parameter option
        :param I: Matrix of 1's
        :param Nc: Number of ports on which data is being fit
        :param SER1: Current (perturbed) model
        :param oldT0: Eigenvectors from pervious iteration
        :param s_pass: Frequencies being sampled
        :param EE1: Matrix of eigenvalues
        :return: EE1: filled in; oldT0 updated with the last k value
        """
        for k in range(s_pass.shape[0]):
            Y = self.calculate_y_from_SER(I, SER1, k, s_pass)  # Implements Eq. (15) in [6]
            EE1, oldT0 = self.calculate_eigenvalues_for_EE(EE1, Nc, Y, k, oldT0)
        return EE1, oldT0

    def plot_eigenvalues_of_gs(self, s_pass, EE0, xlimflag, ylimflag):
        """
        Plots eigenvalues of G(s)
        :param s_pass: Frequencies being sampled
        :param EE0: Eigenvalues of the current model
        :param xlimflag: Indicates xlim should be used
        :param ylimflag: Indicates ylim should be used
        :return:
        """
        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'

        fig = plt.figure(7, figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1)
        freq = (s_pass / (2 * np.pi * 1j)).real
        ax.plot(freq, (EE0.T).real, color='b', linewidth=1)
        if xlimflag:
            plt.xlim(freq[0], self.options['xlim'])
        else:
            plt.xlim(freq[0], freq[-1])
        if np.any(ylimflag):
            plt.ylim(self.options['ylim'][0], self.options['ylim'][-1])
        plt.xlim(freq[0], freq[-1])

        plt.xlabel('Frequency [Hz]')
        if self.options['parametertype'] == ParameterType.y:
            plt.ylabel('Eigenvalues of G')
        else:
            plt.ylabel('Eigenvalues of S')
        plt.title("Eigenvalues of G(s)")
        plt.tight_layout()
        plt.savefig('eigenvalues_of_Gs')
        plt.show()
        return


    def plot_figure_7(self, s_pass, EE1, xlimflag, ylimflag):
        """
        Plots eigenvalues of G(s)
        :param s_pass: Frequencies being sampled
        :param EE1: Eigenvalues of the perturbed model
        :param xlimflag: Indicates xlim should be used
        :param ylimflag: Indicates ylim should be used
        :return:
        """
        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'

        fig = plt.figure(7, figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1)
        freq = (s_pass / (2 * np.pi * 1j)).real
        ax.plot(freq, (EE1.T).real, color='r', linewidth=1, label='EE1')
        if xlimflag:
            plt.xlim(freq[0], self.options['xlim'])
        else:
            plt.xlim(freq[0], freq[-1])
        if np.any(ylimflag):
            plt.ylim(self.options['ylim'][0], self.options['ylim'][-1])
        plt.xlim(freq[0], freq[-1])

        plt.xlabel('Frequency [Hz]')
        if self.options['parametertype'] == ParameterType.y:
            plt.ylabel('Eigenvalues of G')
        else:
            plt.ylabel('Eigenvalues of S')
        plt.title("Eigenvalues of G")
        plt.legend()
        plt.tight_layout()
        plt.savefig('Eigenvalues_of_G_end')
        plt.show()
        return

    def plot_figure_8(self, s_pass, EE0, EE1, xlimflag, ylimflag):
        """
        Plots eigenvalues of G(s) of the current vs. perturbed model
        :param s_pass: Frequencies being sampled
        :param EE0: Eigenvalues of the current model
        :param EE1: Eigenvalues of the perturbed model
        :param xlimflag: Indicates xlim should be used
        :param ylimflag: Indicates ylim should be used
        :return:
        """

        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'

        fig = plt.figure(8, figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1)
        freq = (s_pass / (2 * np.pi * 1j)).real
        ax.plot(freq, (EE0.T).real, color='b', linewidth=1, label='Previous')
        ax.plot(freq, (EE1.T).real, color='r', linewidth=1, label='Perturbed')
        if xlimflag:
            plt.xlim(freq[0], self.options['xlim'])
        else:
            plt.xlim(freq[0], freq[-1])
        if np.any(ylimflag):
            plt.ylim(self.options['ylim'][0], self.options['ylim'][-1])
        plt.xlim(freq[0], freq[-1])

        plt.xlabel('Frequency [Hz]')
        if self.options['parametertype'] == ParameterType.y:
            plt.ylabel('Eigenvalues of G')
        else:
            plt.ylabel('Eigenvalues of S')
        plt.title("Monitoring enforcement process")
        plt.legend()
        plt.tight_layout()
        plt.savefig('Monitoring_enforcement_process')
        plt.show()
        return