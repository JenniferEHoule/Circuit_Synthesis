""" VFdriver.py

Author: Jennifer Houle
Date: 3/13/2020

This program is based off VFdriver.m from [4]. From [4],

PURPOSE : Calculate a rational model of a symmetric matrix of frequency domain
          data (s,H(s)), on pole-residue form, and on state space form

             H(s)=SUM(Rm/(s-am)) +D +s*E    %pole-residue
                       m

             H(s)=C*(s*I-A)^(-1)*B +D +s*E  %state-space

APPROACH: The elements of the upper triangle of H are stacked into a single vector
          that is fitted using a Fast implementation of the Relaxed version of Vector Fitting
          (FRVF) as implemented in vectfit3.m. All matrix elements are fitted with a common
          pole set.


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

"""
from enum import Enum, auto
from math import pi, sqrt, log10, ceil, floor

import scipy.linalg
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

from vectfit3 import VectorFit3
from utils import AsympOptions, WeightParam


class PoleType(Enum):
    """
    This class contains the options for poletype.
    'lincmplx' : linearly spaced, complex conjugate pairs
    'logcmplx' : logarithmically spaced, complex conjugate pairs
    """
    lincmplx = 'lincmplx'
    logcmplx = 'logcmplx'
    linlogcmplx = 'linlogcmplx'

class VFdriver:
    """
    This implements the Vector Fitting algorithm
    """
    DEFAULT_OPTIONS = dict(
        N=1,
        poletype=PoleType.lincmplx,
        nu=1e-3,
        Niter1=4,
        Niter2=4,
        weight=None,
        weightparam=WeightParam.common_1,
        asymp=AsympOptions.D,
        stable=True,
        relaxed=True,
        plot=True,
        logx=True,
        logy=True,
        errplot=True,
        phaseplot=True,
        screen=True,
        cmplx_ss=True,
        remove_HFpoles=False,
        factor_HF=1.1,
        passive_DE=False,
        passive_DE_TOLD=1e-6,
        passive_DE_TOLE=1e-16
    )

    VF_options = dict(
        stable = True,
        relax = True,
        spy2 = True,
        logx = True,
        logy = True,
        errplot = True,
        phaseplot = True,
        cmplx_ss = True
    )

    def __init__(self, **options):
        """
        Sets up the options by merging the default options with any the user selects.
        User selected options have priority
        """
        if 'poletype' in options:
            options['poletype'] = PoleType(options['poletype'])
        if 'weightparam' in options:
            options['weightparam'] = WeightParam(options['weightparam'])
        if 'asymp' in options:
            options['asymp'] = AsympOptions(options['asymp'])
        self.options = {**self.DEFAULT_OPTIONS, **options}

        self.vf_opts = self.VF_options
        self.vf_opts['stable'] = self.options['stable']
        self.vf_opts['relax'] = self.options['relaxed']
        self.vf_opts['spy2'] = self.options['plot']
        self.vf_opts['logx'] = self.options['logx']
        self.vf_opts['logy'] = self.options['logy']
        self.vf_opts['errplot'] = self.options['errplot']
        self.vf_opts['phaseplot'] = self.options['phaseplot']
        self.vf_opts['cmplx_ss'] = self.options['cmplx_ss']


    def vfdriver(self, bigH, s, poles=None):
        """
        """

        Ns = s.shape[0]

        if poles is None:
            if self.options['N'] == 1:
                print("Note: N was set to 1; check this was intentional")
                return
            N = self.options['N']
            oldpoletype = self.options['poletype']
            if N < 6:
                if self.options['poletype'] == PoleType.linlogcmplx:
                    self.options['poletype'] = PoleType.logcmplx
            nu = self.options['nu']

            if self.options['poletype'] == PoleType.logcmplx or \
                    self.options['poletype'] ==PoleType.logcmplx:
                if self.options['poletype'] == PoleType.lincmplx:
                    bet = np.linspace(s[0] / 1j, s[Ns - 1], N // 2)
                    poles = np.zeros(bet.shape[0], dtype=complex)
                    for n in range(bet.shape[0]):
                        alf = -nu * bet[n]
                        poles[n * 2] = alf - 1j * bet[n]
                        poles[n * 2 + 1] = alf + 1j * bet[n]
                elif self.options['poletype'] == PoleType.logcmplx:
                    bet = np.logspace(log10((s[0] / 1j).real), log10((s[Ns - 1] / 1j).real), num=floor(N / 2))
                    poles = np.zeros(bet.shape[0] * 2, dtype=complex)
                    for n in range(bet.shape[0]):
                        alf = -nu * bet[n]
                        poles[n * 2] = alf - 1j * bet[n]
                        poles[n * 2 + 1] = alf + 1j * bet[n]
                else:
                    print(f"ERROR in poleresiduefit.py: Illegal value for options.poletype/n"
                          "Valid input: ''lincmplex'' and ''logcmplx'''\n"
                          "Given input: {self.options['poletype']}")
                    return
            elif self.options['poletype'] == PoleType.linlogcmplx:
                bet = np.linspace(s[0] / 1j, s[Ns - 1] / 1j, ceil((N - 1) / 4), dtype=complex)
                poles1 = np.zeros(bet.shape[0] * 2, dtype=complex)
                for n in range(bet.shape[0]):
                    alf = -nu * bet[n]
                    poles1[n * 2] = alf - 1j * bet[n]
                    poles1[n * 2 + 1] = alf + 1j * bet[n]
                bet = np.logspace(log10(s[0].imag), log10(s[Ns - 1].imag), 2 + floor(N / 4))
                bet = np.delete(bet, [0])
                bet = np.delete(bet, [bet.shape[0] - 1])
                poles2 = np.zeros(bet.shape[0] * 2, dtype=complex)
                for n in range(0, bet.shape[0]):
                    alf = -nu * bet[n]
                    poles2[n * 2] = alf - 1j * bet[n]
                    poles2[n * 2 + 1] = alf + 1j * bet[n]
                poles = np.append(poles1, poles2)
            else:
                print("Invalid option for Pole Type")
                return

            if poles.shape[0] < N:
                if self.options['poletype'] == PoleType.lincmplx:
                    pole_extra = -(s[0] / 1j + s[-1] / 1j) / 2  # Placing surplus pole in midpoint
                elif self.options['poletype'] == PoleType.logcmplx or self.options['poletype'] == PoleType.linlogcmplx:
                    pole_extra = -10 ** (log10(s[0] / 1j) + log10(s[-1] / 1j)) / 2  # Placing surplus pole at midpoint
                poles = np.append(poles, pole_extra)
            self.options['poletype'] = oldpoletype

        Nc = bigH[:, 0, 0].shape[0]
        Ns = s.shape[0]

        if self.options['screen'] == True:
            print("START: ")
            print("Stacking matrix elements (lower triangle) into single column")
        tell = 0
        f_string = []
        for col in range(0, Nc):
            for row in range(col, Nc):
                tell = tell + 1
                f_string.append(bigH[row, col, :])
        f = np.vstack(f_string)
        nnn = tell

        # Fitting options
        self.vf_opts['spy1'] = False
        self.vf_opts['skip_pole'] = False
        self.vf_opts['skip_res'] = True
        self.vf_opts['legend'] = True

        oldspy2 = self.vf_opts['spy2']
        self.vf_opts['spy2'] = False

        if Nc == 1:
            f_sum = f
        if Nc > 1: # Will do only for multi-terminal case
            # Forming columns sum and associated LS weight:
            f_sum = 0
            tell = 0
            for row in range(0, Nc):
                for col in range(row, Nc):
                    if self.options['weightparam'] == WeightParam.common_1 or self.options['weightparam'] == WeightParam.common_norm or self.options['weightparam'] == WeightParam.common_sqrt:
                        f_sum = f_sum + f[tell, :]
                    elif self.options['weightparam'] == WeightParam.indiv_norm:
                        f_sum = f_sum + f[tell, :] / LA.norm(f[tell, :])
                    elif self.options['weightparam'] == WeightParam.indiv_sqrt:
                        f_sum = f_sum + f[tell, :] / sqrt(LA.norm(f[tell, :]))
                    tell = tell + 1

        # Creating LS weight
        if self.options['weight'] == None:
            weight = np.zeros((1, Ns))
            if self.options['weightparam'] == WeightParam.common_1:
                weight = np.ones((1, Ns))
                weight_sum = np.ones((1, Ns))
            elif self.options['weightparam'] == WeightParam.indiv_norm:
                weight[0, :] = 1/np.abs(f)
                weight_sum = 1/np.abs(f_sum)
            elif self.options['weightparam'] == WeightParam.indiv_sqrt:
                weight[0, :] = 1/np.sqrt(np.abs(f))
                weight_sum = 1/ np.sqrt(np.abs(f_sum))
            elif self.options['weightparam'] == WeightParam.common_norm:
                for k in range(Ns):
                    weight[0, k] = 1/LA.norm(f[:, k])
                weight_sum = weight
            elif self.options['weightparam'] == WeightParam.common_sqrt:
                for k in range(Ns):
                    weight[0, k] = 1/np.sqrt(LA.norm(f[:, k]))
                weight_sum = weight
            else:
                print(f"ERROR in VFdriver; invalid option for 'weight': {self.options['weightparam']}.")
        else:
            weight = np.zeros((nnn, Ns))
            tell = 0
            for row in range(Nc):
                for col in range(row, Nc):
                    weight[tell, :] = self.options['weight'][row, col, :]
            weight_sum = np.ones((1, Ns))

        if Nc > 1:  # Will do only for multi-terminal case
            if self.options['screen'] == True:
                print("Calculating improved initial poles by fitting column sum")
            for iter in range(self.options['Niter1']):
                if self.options['screen'] == True:
                    print(f'\tIteration : {iter}')
                vector_fitter = VectorFit3(stable=self.vf_opts['stable'],
                                            relax=self.vf_opts['relax'],
                                            spy2=self.vf_opts['spy2'],
                                            logx=self.vf_opts['logx'],
                                            logy=self.vf_opts['logy'],
                                            errplot=self.vf_opts['errplot'],
                                            phaseplot=self.vf_opts['phaseplot'],
                                            cmplx_ss=self.vf_opts['cmplx_ss'])
                SER, poles, rmserr, fit = vector_fitter.do_fit(f_sum.reshape(1, -1), s, poles.reshape(1, -1), weight_sum)

        if self.options['screen'] == True:
            print("Fitting column")
        self.vf_opts['skip_res'] = True
        for iter in range(self.options['Niter2']):
            if self.options['screen'] == True:
                print(f'\tIteration : {iter}')
            if iter == self.options['Niter2'] - 1:
                self.vf_opts['skip_res'] = False
            vector_fitter = VectorFit3(stable=self.vf_opts['stable'],
                                       skip_res=self.vf_opts['skip_res'],
                                       asymp=self.options['asymp'],
                                       relax=self.vf_opts['relax'],
                                       spy2=self.vf_opts['spy2'],
                                       logx=self.vf_opts['logx'],
                                       logy=self.vf_opts['logy'],
                                       errplot=self.vf_opts['errplot'],
                                       phaseplot=self.vf_opts['phaseplot'],
                                       cmplx_ss=self.vf_opts['cmplx_ss'])
            SER, poles, rmserr, fit1 = vector_fitter.do_fit(f, s, poles.reshape(1, -1), weight)
        if self.options['Niter2'] == 0:
            self.vf_opts['skip_res'] = False
            self.vf_opts['skip_pole'] = True
            vector_fitter = VectorFit3(stable=self.vf_opts['stable'],
                                       skip_res=self.vf_opts['skip_res'],
                                       skip_pole=self.vf_opts['skip_pole'],
                                       asymp=self.options['asymp'],
                                       relax=self.vf_opts['relax'],
                                       spy2=self.vf_opts['spy2'],
                                       logx=self.vf_opts['logx'],
                                       logy=self.vf_opts['logy'],
                                       errplot=self.vf_opts['errplot'],
                                       phaseplot=self.vf_opts['phaseplot'],
                                       cmplx_ss=self.vf_opts['cmplx_ss'])
            SER, poles, rmserr, fit1 = vector_fitter.do_fit(f, s, poles.reshape(1, -1), weight)

        # Throwing out high-frequency poles
        fit2 = fit1
        if self.options['remove_HFpoles'] == True:
            if self.options['screen'] == True:
                print("Throwing out high-frequency poles")
            poles = poles[np.nonzero(np.abs(poles) < self.options['factor_HF'] * np.abs(s[-1]))]
            N = poles.shape[0]
            if self.options['screen'] == True:
                print("Refitting residues")
            self.vf_opts['skip_pole'] = True
            vector_fitter = VectorFit3(stable=self.vf_opts['stable'],
                                       skip_res=self.vf_opts['skip_res'],
                                       skip_pole=self.vf_opts['skip_pole'],
                                       asymp=self.options['asymp'],
                                       relax=self.vf_opts['relax'],
                                       spy2=self.vf_opts['spy2'],
                                       logx=self.vf_opts['logx'],
                                       logy=self.vf_opts['logy'],
                                       errplot=self.vf_opts['errplot'],
                                       phaseplot=self.vf_opts['phaseplot'],
                                       cmplx_ss=self.vf_opts['cmplx_ss'])
            SER, poles, rmserr, fit2 = vector_fitter.do_fit(fit1, s, poles.reshape(1, -1), weight)

        fit3 = None
        if self.options['passive_DE'] == True and self.options['asymp'] != AsympOptions.NONE:
            if self.options['screen'] == True:
                if self.options['asymp'] == AsympOptions.D:
                    print("Enforcing positive realness for D")
                elif self.options['asymp'] == AsympOptions.DE:
                    print("Enforcing positive realness for D, E")
            tell = 0
            DD = np.zeros(Nc)
            EE = np.zeros(Nc)
            for col in range(Nc):
                for row in range(col, Nc):
                    DD[row, col] = SER['D'][tell]
                    EE[row,col] = SER['E'][tell]
                    tell = tell + 1
            DD = DD + np.tril(DD, -1)
            EE = EE + np.tril(EE, -1)

            # Calculating Dmod, Emod:
            V, L = LA.eigvals(DD)
            for n in range(Nc):
                if L[n, n] < 0:
                    L[n, n] = self.options['passive_DE_TOLD']
            DD = (V.dot(L)).dot(LA.matrix_power(V, -1))
            V, L = LA.eigvals(EE)
            for n in range(Nc):
                if L[n, n] < 0:
                    L[n, n] = self.options['passive_DE_TOLE']
            EE = (V.dot(L)).dot(LA.matrix_power(V, -1))
            tell = 0

            #Calculating fmod:
            Emod = np.zeros((Nc, 1))
            Dmod = np.zeros((Nc, 1))
            fmod = np.zeros((Nc, fit.shape[1]))
            for col in range(Nc):
                for row in range(col, Nc):
                    Dmod[tell] = DD[row, col]
                    Emod[tell] = EE[row, col]
                    fmod[tell, :] = fit[tell, :] - Dmod[tell] - s * Emod[tell]
                    tell = tell + 1
            if self.options['screen'] == True:
                if self.options['asymp'] == AsympOptions.D:
                    print("Refitting C while enforcing D = 0")
                elif self.options['asymp'] == AsympOptions.DE:
                    print("Refitting C while enforcing D = 0, E = 0")
            self.vf_opts['skip_pole'] = False
            self.options['asymp'] = AsympOptions.NONE
            vector_fitter = VectorFit3(stable=self.vf_opts['stable'],
                                       skip_res=self.vf_opts['skip_res'],
                                       skip_pole=self.vf_opts['skip_pole'],
                                       asymp=self.options['asymp'],
                                       relax=self.vf_opts['relax'],
                                       spy2=self.vf_opts['spy2'],
                                       logx=self.vf_opts['logx'],
                                       logy=self.vf_opts['logy'],
                                       errplot=self.vf_opts['errplot'],
                                       phaseplot=self.vf_opts['phaseplot'],
                                       cmplx_ss=self.vf_opts['cmplx_ss'])
            SER, poles, rmserr, fit3 = vector_fitter.do_fit(fit1, s, poles.reshape(1, -1), weight)
            SER['D'] = Dmod
            SER['E'] = Emod
            for tell in range(fit3.shape[0]):
                fit3[tell, :] = fit3[tell, :] + SER['D'] + s * SER['E']
        if Nc > 1:
            if self.options['screen'] == True:
                print("Transforming model of lower matrix triangle into state-space model of full matrix")
            SER = self.tri2full(SER)
        if self.options['screen'] == True:
            print("Generating pole-residue model")
        R, a = self.ss2pr(SER['A'], SER['B'], SER['C'])
        SER['R'] = R.copy()
        SER['poles'] = a.copy()

        #RMS Error of Fitting:
        if fit3:
            if(fit3.shape[0] != 0):
                fit = fit3
        elif fit2.shape[0] != 0:
            if(fit2.shape[0] != 0):
                fit = fit2
        elif fit1.shape[0] != 0:
            if(fit1.shape[0] != 0):
                fit = fit1

        diff = np.swapaxes(fit, 0, 1) - f
        rmserr = sqrt(np.sum(np.sum(np.abs(diff ** 2)))) / sqrt(nnn * Ns)

        self.vf_opts['spy2'] = oldspy2
        if self.vf_opts['spy2'] == True:
            if self.options['screen'] == True:
                print("Plotting of results")
            self.plot_magnitude_and_phase(fit, f, s, Nc, Ns)

        if self.options['screen'] == True:
            print("End")

        bigHfit = np.zeros((Nc, Nc, Ns), dtype=complex)
        tell = 0
        fit = np.swapaxes(fit, 0, 1)
        for row in range(Nc):
            for col in range(row, Nc):
                bigHfit[row, col, :] = fit[tell, :]
                if row != col:
                    bigHfit[col, row,:] = fit[tell, :]
                tell = tell + 1

        return SER, rmserr, bigHfit

    def tri2full(self, SER):
        """

        :param SER: state space model (A - diagonal matrix, B - vector of 1's, C - matrix of residues per element,
        D - vector of D values, E - vector of E value are relevant to this function)
        :return: SER2: reshaped state space model
        """
        A = SER['A']
        B = SER['B']
        C = SER['C']
        D = SER['D']
        E = SER['E']

        tell = 0
        Nc = 1
        for k in range(1, 10000):
            tell = tell + k
            if tell == D.shape[0]:
                Nc = k
                break

        N = A.shape[0]
        tell = 0
        AA = np.zeros((0, 0), dtype=complex)
        BB = np.zeros((0, 0), dtype=complex)
        CC = np.zeros((Nc, Nc * N), dtype=complex)
        DD = np.zeros((Nc, Nc), dtype=complex)
        EE = np.zeros((Nc, Nc), dtype=complex)
        for col in range(Nc):
            AA = scipy.linalg.block_diag(AA, A)
            BB = scipy.linalg.block_diag(BB, B)
            if len(D.shape) == 1:
                D = D.reshape(-1, 1)
            for row in range(col, Nc):
                DD[row, col] = D[tell, 0]
                EE[row, col] = E[tell, 0]
                CC[row, col * N:(col + 1) * N] = C[tell, :]
                CC[col, row * N:(row + 1) * N] = C[tell, :]
                tell = tell + 1
        DD = DD + (DD - np.diagflat(np.diag(DD))).T
        EE = EE + (EE - np.diagflat(np.diag(EE))).T

        SER2 = dict(
            A=AA.copy(),
            B=BB.copy(),
            C=CC.copy(),
            D=DD.copy(),
            E=EE.copy()
        )

        return SER2

    def ss2pr(self, A, B, C):

        # Converting real-only state-space model into complex model, if necessary
        if np.max(np.max(np.abs(A - np.diag(np.diag(A))))) != 0:
            errflag = 0
            for m in range(A.shape[0]):
                if A[m, m + 1] != 0:
                    A[m, m] = A[m, m] + 1j * A[m, m + 1]
                    A[m + 1, m + 1] = A[m + 1, m + 1] - 1j * A[m, m + 1]

                    B[m, :] = (B[m, :] + B[m + 1, :]) / 2
                    B[m + 1, :] = B[m, :].copy()

                    C[:, m] = C[:, m] + 1j * C[:, m + 1]
                    C[:, m + 1] = np.conj(C[:, m])
        # Converting complex state-space model into pole-residue model
        Nc = C.shape[0]
        N = int(A.shape[0] / Nc)
        R = np.zeros((Nc, Nc, N), dtype=complex)
        for m in range(N):
            Rdum = np.zeros(Nc)
            for n in range(Nc):
                ind = n * N + m
                Rdum = Rdum + C[:, ind].reshape(-1,1) @ B[ind, :].reshape(1,-1)
            R[:, :, m] = Rdum.copy()
        a = np.diag(A[0:N, 0:N])
        return R.copy(), a.copy()

    def plot_magnitude_and_phase(self, fit, f, s, Nc, Ns):
        """
        Plot the magnitude and the phase (if desired) of input f data and new fit data across the input frequencies.
        Repeat for each port.
        :param fit: This is created from the new poles and the calculated residues
        :return:
        """
        freq = s / (2 * pi * 1j)
        freq = freq.reshape(1, -1)

        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'

        number_of_plots = f.shape[0]

        for plot_num in range(0, number_of_plots):
            fig = plt.figure(1, figsize=(8, 7))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(freq[0, :].real, np.abs(f[plot_num, :]), color='b', linewidth=1,
                    label='Data {}'.format(plot_num))
            ax.plot(freq[0, :].real, np.abs(fit[:, plot_num]), color='r', linewidth=1, label='FRVF {}'.format(plot_num))
            if self.vf_opts['errplot']:
                ax.plot(freq[0, :].real, np.abs(f[0] - fit.T[0]), color='g', linewidth=1, label='Deviation')
            plt.xlim(freq[0, 0].real, freq[0, Ns - 1].real)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            self.set_scale_legend_options()
            plt.tight_layout()
            plt.savefig('magnitude_plot_{}'.format(plot_num))
            plt.show()
        if self.vf_opts['phaseplot']:
            for plot_num in range(number_of_plots):
                fig = plt.figure(1, figsize=(8, 7))
                ax = fig.add_subplot(1, 1, 1)
                plt.xlim(freq[0, 0].real, freq[0, Ns - 1].real)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Phase Angle [deg]')
                ax.plot(freq[0, :].real, 180 / pi * np.unwrap(np.angle(f[plot_num, :])), color='b', linewidth=1,
                        label='Data {}'.format(plot_num))
                ax.plot(freq[0, :].real, 180 / pi * np.unwrap(np.angle(fit[:, plot_num])), color='r', linewidth=1,
                        label='FRVF {}'.format(plot_num))
                if self.vf_opts['errplot']:
                    ax.plot(freq[0, :].real, np.abs((180 / pi * np.unwrap(np.angle(f[plot_num, :]))) -
                                                    (180 / pi * np.unwrap(np.angle(fit[:, plot_num])))),
                            color='g', linewidth=1,
                            label='Deviation')
                self.set_scale_legend_options()
                plt.yscale('linear')  # Force linear scale for phase plot
                plt.tight_layout()
                plt.savefig('phase_plot_{}'.format(plot_num))
                plt.show()

    def set_scale_legend_options(self):
        """
        This sets the logarithmic scaling and legend options.
        :return:
        """
        if self.vf_opts['logx']:
            if self.vf_opts['logy']:
                plt.xscale('log')
                plt.yscale('log')
            else:
                plt.xscale('log')
        else:
            if self.vf_opts['logy']:
                plt.yscale('log')
        if self.vf_opts['legend']:
            plt.legend()




