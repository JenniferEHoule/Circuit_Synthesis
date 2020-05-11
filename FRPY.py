""" FRPY.py

Author: Jennifer Houle
Date: 3/30/2020

This program is based off FRPY.m functions from [4]. From [4],

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


import numpy as np
import scipy.linalg as LA
import numba

from quadprog import quadprog
from fitcalc import fitcalcPRE
from pr2ss import pr2ss
from utils import PoleTypes, find_which_poles_are_complex, OutputLevel, WeightParam, sort_eigenvalues_eigenvectors_to_match_matlab


def FRPY(SER, s, s2, s3, RPopts):

    # Reshape the s data to be a matrix (v. a vector)
    s2 = s2.reshape(-1, 1).copy()

    auxflag = True      # I don't see an option for this to not be true in the MATLAB code so I'm hard-coding it

    # Create copies of input parameters / data
    weightfactor = RPopts['weightfactor']
    weightparam = RPopts['weightparam']
    if RPopts['weight']:
        bigweight = RPopts['weight']
    TOLE = RPopts['TOLE']
    TOL = RPopts['TOLGD']

    SERA = SER['poles'].reshape(-1, 1).copy()
    SERC = SER['R'].copy()
    SERD = SER['D'].copy()
    SERE = SER['E'].copy()

    m = SERA.shape[0]
    n = SERA.shape[1]

    if m < n:
        SERA = SERA.T

    if not np.any(RPopts.get('H')):
        RPopts['H'] = []
        RPopts['oldDflag'] = -1
        RPopts['oldEflag'] = -1

    # This is used for checking Eq. (4), (5) in [6]
    Dflag, VD, eigD, invVD = create_eig_n(SERD)
    Eflag, VE, eigE, invVE = create_eig_n(SERE)

    SERCnew = SERC
    SERDnew = SERD
    SEREnew = SERE

    N = SERA.shape[0]


    Ns = s.shape[0]
    Ns2 = s2.shape[0]
    Ns3 = s3.shape[0]
    Nc = SERD.shape[0]
    Nc2 = Nc * Nc

    # LS problem:
    # Finding out which poles are complex:
    cindex = find_which_poles_are_complex(N, np.diagflat(SERA))

    # Build Asys as seen in Eq. (9a) in [6]
    if not np.any(RPopts.get('H')):
        if RPopts['outputlevel'] == OutputLevel.max:
            print("Building system equation (once)")

        # This sets the size of bigA based on whether D and E must be perturbed
        if Dflag + Eflag == 2:
            bigA = np.zeros((Ns * Nc2, Nc * (N + 2)), dtype=complex)
        elif Dflag + Eflag == 1:
            bigA = np.zeros((Ns * Nc2, Nc * (N + 1)), dtype=complex)
        else:
            bigA = np.zeros((Ns * Nc2, Nc * N), dtype=complex)

        # Setting up some of the matrices
        bigV = np.zeros((Nc, Nc * N))
        biginvV = np.zeros((Nc, Nc * N))
        bigD = np.zeros((Nc, N))

        for m in range(N):
            R = SERC[:, :, m]
            # Change the residues to real based on which type of pole it is
            if cindex[m] == PoleTypes.REAL:
                R = R
            elif cindex[m] == PoleTypes.COMPLEX_FIRST:
                R = R.real
            else:
                R = R.imag
            D, V = LA.eig(R)        # The eigenvalues / vectors differ from Octave but both should be valid
            D, V = sort_eigenvalues_eigenvectors_to_match_matlab(D, V)

            bigV[:Nc, m * Nc:(m + 1) * Nc] = V.real.copy()
            biginvV[:Nc, m * Nc:(m + 1) * Nc] = np.linalg.matrix_power(V.real, -1)
            bigD[:, m] = D.real

        for k in range(Ns):
            sk = s[k].copy()

            # Calculating matrix Mmat (coefficients for LS-problem)
            Yfit = fitcalcPRE(sk.reshape(1, -1), SERA, SERC, SERD, SERE)

            if weightparam:
                weight = calculate_weight(weightparam, Nc, Yfit)
            else:
                weight = bigweight[:, :, k]

            bigA, bigA2 = fill_in_bigA(N, bigV, Nc, sk, cindex, SERA, weight, Dflag, Eflag, VD, invVD, VE, invVE, bigA, np.array([]), k, Nc2)

        # Introducing Samples Outside LS Region: One Sample per Pole (s4)
        s4 = np.array([])
        for m in range(N):
            if cindex[m] == PoleTypes.REAL:
                if (np.abs(SERA[m]) > s[Ns - 1] / 1j) or (np.abs(SERA[m]) < s[0] / 1j):
                    s4 = np.append(s4, 1j * np.abs(SERA[m]))
            elif cindex[m] == PoleTypes.COMPLEX_FIRST:
                if (np.abs(SERA[m].imag) > s[Ns - 1] / 1j) or (np.abs(SERA[m].imag) < s[0] / 1j):
                    s4 = np.append(s4, 1j * np.abs(SERA[m].imag))
        Ns4 = s4.shape[0]

        bigA2 = np.zeros((Ns4 * Nc2, Nc * (N + Dflag + Eflag)), dtype=complex)

        for k in range(Ns4):
            sk = s4[k]
            # Calculating matrix Mmat (coefficients for LS-problem)
            tell = 0
            offs = 0
            Yfit = fitcalcPRE(sk.reshape(1, -1), SERA, SERC, SERD, SERE)

            weight = calculate_weight(weightparam, Nc, Yfit)
            weight = weight * weightfactor

            bigA, bigA2 = fill_in_bigA(N, bigV, Nc, sk, cindex, SERA, weight, Dflag, Eflag, VD, invVD, VE, invVE, bigA, bigA2, k, Nc2)

        bigA = np.vstack((bigA, bigA2))
        bigA = np.vstack((bigA.real, bigA.imag))
        Acol = bigA.shape[1]
        Escale = np.zeros(Acol)
        for col in range(Acol):
            Escale[col] = LA.svd(bigA[:, col].reshape(-1, 1))[1]
            bigA[:, col] = bigA[:, col] / Escale[col]

        H = bigA.T @ bigA
        RPopts['H'] = H
        RPopts['Escale'] = Escale
        RPopts['bigV'] = bigV
        RPopts['biginvV'] = biginvV
        if RPopts['outputlevel'] == OutputLevel.max:
            print("Done")
    else:
        bigV = RPopts['bigV']
        biginvV = RPopts['biginvV']
        if Dflag != RPopts['oldDflag'] or Eflag != RPopts['oldEflag']:
            RPopts['H'] = RPopts['H'][:Nc * (N + Dflag + Eflag), :Nc * (N + Dflag + Eflag)]
            RPopts['Escale'] = RPopts['Escale'][:Nc * (N + Dflag + Eflag)]

    viol_G = np.array([])
    viol_D = np.array([])
    viol_E = np.array([])

    bigB = np.empty((0, Nc * (N + Dflag + Eflag)))
    bigc = np.array([])

    # Loop for constraint problem, Type #1 (violating eigenvalues in s2):
    viol_G, bigB, bigc = generate_violG(Nc, SERA, SERC, SERD, SERE, N, Ns2, s2, bigV, biginvV, TOL, cindex, Dflag, Eflag, VD, invVD, viol_G, Nc2, bigB, bigc)

    # Loop for Constraint Problem, Type #2 (all eigenvalues in s3)
    viol_G, bigB, bigc = generate_violG(Nc, SERA, SERC, SERD, SERE, N, Ns3, s3, bigV, biginvV, TOL, cindex, Dflag, Eflag, VD, invVD, viol_G, Nc2, bigB, bigc)

    # Adds constraint for the event D < 0
    if Dflag == 1:
        for n in range(Nc):
            dum = np.zeros((1, Nc * (N + Dflag + Eflag)))
            dum[0, Nc * N + n] = 1
            bigB = np.vstack((bigB, dum))
            bigc = np.append(bigc, np.diag(eigD)[n] - TOL)
            viol_G = np.append(viol_G, np.diag(eigD)[n])
            viol_D = np.append(viol_D, np.diag(eigD)[n])

    # Adds constraint for event E < 0
    if Eflag == 1:
        for n in range(Nc):
            dum = np.zeros((1, Nc * (N + Dflag + Eflag)))
            dum[0, Nc * (N + Dflag) + n] = 1
            bigB = np.vstack((bigB, dum))
            bigc = np.append(bigc, eigE[n] - TOLE)
            viol_E = np.append(viol_E, eigE[n])

    if bigB.shape[0] == 0:
        return SER, RPopts # No passivity violations

    bigB = bigB.real
    for col in range(RPopts['H'].shape[0]):
        if bigB.shape[0] > 0:
            bigB[:, col] = bigB[:, col] / RPopts['Escale'][col]

    ff = np.zeros(RPopts['H'].shape[0])

    print("Starting quadprog...")

    # Solve Eq. (9a), (9b) in [6]
    dx = quadprog(RPopts['H'], ff, -bigB, bigc.real)
    dx = dx / RPopts['Escale'].T

    # New state space model created (see Eq. (8a), (8b), (8c) in [6])
    for m in range(N):
        if cindex[m] == PoleTypes.REAL:
            D1 = np.diagflat(dx[m * Nc: (m + 1) * Nc].copy())
            SERCnew[:, :, m] = SERC[:, :, m] + bigV[:, m * Nc:(m + 1) * Nc] @ D1 @ biginvV[:, m * Nc:(m + 1) * Nc]
        elif cindex[m] == PoleTypes.COMPLEX_FIRST:
            GAMM1 = bigV[:, m * Nc:(m + 1) * Nc].copy()
            GAMM2 = bigV[:, (m + 1) * Nc: (m + 2) * Nc].copy()
            invGAMM1 = biginvV[:, m * Nc:(m + 1) * Nc].copy()
            invGAMM2 = biginvV[:, (m + 1) * Nc:(m + 2) * Nc].copy()

            D1 = np.diagflat(dx[m * Nc:(m + 1) * Nc]).copy()
            D2 = np.diagflat(dx[(m + 1) * Nc: (m + 2) * Nc]).copy()
            R1 = SERC[:, :, m].real.copy()
            R2 = SERC[:, :, m].imag.copy()
            R1new = R1 + GAMM1 @ D1 @ invGAMM1
            R2new = R2 + GAMM2 @ D2 @ invGAMM2
            SERCnew[:, :, m] = R1new + 1j * R2new
            SERCnew[:, :, m + 1] = R1new - 1j * R2new

    if Dflag == 1:
        DD = np.diagflat(dx[N * Nc:(N + 1) * Nc].copy())
        SERDnew = SERDnew + VD @ DD @ invVD

    if Eflag == 1:
        EE = np.diagflat(dx[(N + Dflag) * Nc: (N + Dflag + Eflag + 1) * Nc].copy())
        SEREnew = SEREnew + VE @ EE @ invVE

    SERDnew = (SERDnew + SERDnew.T) / 2
    SEREnew = (SEREnew + SEREnew.T) / 2
    for m in range(N):
        SERCnew[:, :, m] = (SERCnew[:, :, m] + SERCnew[:, :, m].T) / 2
    SER['R'] = SERCnew.copy()
    SER['D'] = SERDnew.copy()
    SER['E'] = SEREnew.copy()
    SER = pr2ss(SER)

    RPopts['oldDflag'] = Dflag
    RPopts['oldEflag'] = Eflag
    return SER, RPopts

def create_eig_n(SER):
    """
    Used with either SERD or SERE to check for passivity according to Eq. (4) and (5) in [6]
    :param SER: SER will be either SERD or SERE
    :return:
    """
    d, temp = LA.eig(SER) # d are the eigenvalues of the input SER component
    d, temp = sort_eigenvalues_eigenvectors_to_match_matlab(d, temp)
    eigD = d.copy()
    VD = invVD = []
    if np.any(d < 0):   # if any violation of Eq. (4), (5) in [6],
        Dflag = 1 # Will perturb D-matrix (done later)
        eigD, VD = LA.eig(SER)
        eigD, VD = sort_eigenvalues_eigenvectors_to_match_matlab(eigD, VD)
        invVD = np.linalg.matrix_power(VD, -1)
        eigD = np.diag(eigD)
    else:               # Passivity condition met so we don't have to worry about this in the future
        Dflag = 0
    return Dflag, VD, eigD, invVD

def calculate_weight(weightparam, Nc, Yfit):
    if weightparam == WeightParam.common_1:
        weight = np.ones((Nc, Nc))
    elif weightparam == WeightParam.indiv_norm:
        weight = 1 / np.abs(Yfit)
    elif weightparam == WeightParam.indiv_sqrt:
        weight = 1 / np.sqrt(np.abs(Yfit))
    elif weightparam == WeightParam.common_norm:
        weight = np.ones((Nc, Nc)) / LA.norm(np.abs(Yfit))
    elif weightparam == WeightParam.common_sqrt:
        weight = np.ones((Nc, Nc)) / np.sqrt(LA.norm(np.abs(Yfit)))
    else:
        print(f"ERROR! Weight param: {weightparam} invalid!")
        weight = 1
    return weight


def fill_in_bigA(N, bigV, Nc, sk, cindex, SERA, weight, Dflag, Eflag, VD, invVD, VE, invVE, bigA, bigA2, k, Nc2):
    offs = 0

    Mmat = np.zeros((Nc * Nc, bigA.shape[1]), dtype=complex)

    for m in range(N):
        V = bigV[:, m * Nc:(m + 1) * Nc].copy()
        invV = np.linalg.matrix_power(V, -1)

        # See Eq. (A.6) in [1]
        if cindex[m] == PoleTypes.REAL:
            dum = 1 / (sk - SERA[m])
        elif cindex[m] == PoleTypes.COMPLEX_FIRST:
            dum = 1 / (sk-SERA[m]) + 1 / (sk - SERA[m].conj().T)
        else:
            dum = 1j / (sk - SERA[m].conj().T) - 1j / (sk - SERA[m])

        for eigenverdi in range(Nc):
            tell = 0
            gamm = (V[:, eigenverdi]).reshape(-1, 1) @ (invV[eigenverdi, :]).reshape(1, -1)
            for row in range(Nc):
                for col in range(Nc):
                    faktor = weight[row, col]
                    if cindex[m] == PoleTypes.REAL:
                        Mmat[tell, offs + eigenverdi] = gamm[row, col] * faktor * dum
                    elif cindex == PoleTypes.COMPLEX_FIRST:
                        Mmat[tell, offs + eigenverdi] = gamm[row, col] * faktor * dum
                    else:
                        Mmat[tell, offs + eigenverdi] = (gamm[row, col] * faktor * dum)[0]
                    tell = tell + 1
        offs = offs + Nc
    if Dflag:
        Mmat = create_Mmat_from_VD_invVD(Nc, VD, invVD, weight, Mmat, offs)
    if Eflag:
        Mmat = create_Mmat_from_VD_invVD(Nc, VE, invVE, weight, Mmat, offs)

    if bigA2.shape[0] > 0:
        if Eflag:
            Mmat = create_Mmat_from_VD_invVD(Nc, VD, invVD, weight, Mmat, offs)
        bigA2[k * Nc2:(k + 1) * Nc2, :] = Mmat
    else:
        bigA[k * Nc2:(k + 1) * Nc2, :] = Mmat
        bigA2 = []
    return bigA, bigA2


@numba.jit(nopython=True)
def create_Mmat_from_VD_invVD(Nc, VD, invVD, weight, Mmat, offs):
    for eigenverdi in range(Nc):
        gamm = (VD[:, eigenverdi]).reshape(-1, 1) @ (invVD[eigenverdi, :]).reshape(1, -1)
        tell = 0
        for row in range(Nc):
            for col in range(Nc):
                faktor = weight[row, col]
                Mmat[tell, offs + eigenverdi] = gamm[row, col] * faktor
                tell = tell + 1
    return Mmat

def generate_violG(Nc, SERA, SERC, SERD, SERE, N, NsN, sN, bigV, biginvV, TOL, cindex, Dflag, Eflag, VD, invVD, viol_G, Nc2, bigB, bigc):
    """
    This function perturbs the matrix (Eq. (7), (8) in [7])
    """
    Y = np.zeros((Nc, Nc), dtype=complex)
    EE = np.zeros((SERD.shape[0], NsN))
    bigB_size = Nc * (N + Dflag + Eflag)
    sk = np.zeros((1), dtype=complex)
    for k in range(NsN):
        sk[0] = sN[k]
        for row in range(Nc):
            for col in range(Nc):
                Y[row, col] = SERD[row, col] + sk[0] * SERE[row, col]
                Y[row, col] = Y[row, col] + np.sum(SERC[row, col, :] / (sk - SERA[:N]).T)

        # Calculating eigenvalues and eigenvectors:
        Z, V = LA.eig(Y.real)
        Z, V = sort_eigenvalues_eigenvectors_to_match_matlab(Z, V)
        EE[:, k] = Z.real
        if np.amin(Z.real) < 0:  # Any violations
            # Calculating matrix M2mat; matrix of partial derivatives
            tell = 0
            offs = 0

            # Mmat2 will be the top of Eq. (7) in [6]
            Mmat2 = np.zeros((Nc2, bigB_size), dtype=complex)
            for m in range(N):
                VV = bigV[:, m * Nc:(m + 1) * Nc].copy()
                invVV = biginvV[:, m * Nc: (m + 1) * Nc].copy()

                for eigenverdi in range(Nc):
                    tell = 0
                    gamm = VV[:, eigenverdi].reshape(-1, 1) @ invVV[eigenverdi, :].reshape(1, -1)
                    for row in range(Nc):
                        for col in range(Nc):
                            if cindex[m] == PoleTypes.REAL:
                                Mmat2[tell, offs + eigenverdi] = (gamm[row, col] / (sk[0] - SERA[m]))[0]
                            elif cindex[m] == PoleTypes.COMPLEX_FIRST:
                                Mmat2[tell, offs + eigenverdi] = (gamm[row, col] * (1 / (sk[0] - SERA[m]) + 1 / (sk[0] - SERA[m].conj().T)))[0]
                            else:
                                Mmat2[tell, offs + eigenverdi] = (gamm[row, col] * (1j / (sk[0] - SERA[m].conj().T) - 1j / (sk[0] - SERA[m])))[0]
                            tell = tell + 1
                offs = offs + Nc

            if Dflag == 1:
                for eigenverdi in range(Nc):
                    gamm = VD[:, eigenverdi].reshape(-1, 1) @ invVD[eigenverdi, :].reshape(1, -1)

                    tell = 0
                    for row in range(Nc):
                        for col in range(Nc):
                            Mmat2[tell, offs + eigenverdi] = gamm[row, col]
                            tell = tell + 1
            Q = np.zeros((Nc, Nc * Nc), dtype=complex)
            for n in range(Nc):
                tell = 0
                V1 = V[:, n]
                for row in range(Nc):
                    for col in range(Nc):
                        if row == col:
                            qij = V1[row] ** 2
                        else:
                            qij = V1[row] * V1[col]
                        Q[n, tell] = qij
                        tell = tell + 1
            B = Q @ Mmat2   # Eq. (7) in [6]
            delz = Z.real

            for n in range(Nc):  # Unstable?
                if delz[n] < 0:
                    bigB = np.vstack((bigB, B[n, :]))
                    bigc = np.append(bigc, -TOL + delz[n])
                    viol_G = np.append(viol_G, delz[n])
    return viol_G, bigB, bigc