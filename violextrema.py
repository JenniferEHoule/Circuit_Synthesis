""" violextrema.py

Author: Jennifer Houle
Date: 3/29/2020

This program is based off violextremaY.m from [4]. From [4],

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

import numpy as np
import scipy.linalg as LA

from math import pi, log

from fitcalc import fitcalcABCDE
from rot import rot
from intercheig import intercheig


def violextremaY(wintervals, A, C, D, colinterch):
    """
    Identify eigenvalue minima within given intervals
    :param wintervals: The frequency intervals that violate passivity
    :param A: SER[A] - poles
    :param C: SER[C] - residues
    :param D: SER[D] - D values
    :param colinterch: Always true
    :return: s_pass: matrix of frequency intervals where violations occur (each column has the beginning and the end of
        the interval, including 0 and a large value 1e16 if applicable), g_pass: smallest eigenvalue, smin: frequency
        at which g_pass occurred
    """

    if wintervals.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    SERA = A
    SERC = C
    SERD = D

    Nc = SERD.shape[0]
    N = SERA.shape[0]
    A = np.zeros((Nc * N, 1), dtype=complex)
    for col in range(Nc):
        A[col * N: (col + 1) * N, 0] = SERA.reshape(1, -1)
    B = np.zeros((Nc * N, Nc))
    for col in range(Nc):
        B[col * N: (col + 1) * N, col] = np.ones((N))
    C = np.zeros((Nc, Nc * N), dtype=complex)
    for row in range(Nc):
        for col in range(Nc):
            C[row, col * N: (col + 1) * N] = (SERC[row, col, 0:N]).T
    D = SERD
    A = np.diagflat(A)

    s = np.array([])
    Nc = D.shape[0]
    g_pass = 1e16
    smin = 0
    for m in range(wintervals.shape[0]):
        # For each interval, identify global minima for each e-value
        Nint = 21 # number internal frequency samples resolving each interval

        w1 = wintervals[m, 0]
        if wintervals[m, 1] == 1e16: # Violation extends to infinite frequency
            w2 = 2 * pi * 1e16
        else:
            w2 = wintervals[m, 1]

        # Create the frequency interval (combination of linear and logarithmic spacing)
        s_pass1 = 1j * np.linspace(w1, w2, Nint)
        if w1 == 0:
            s_pass2 = 1j * np.logspace(-8, log(w2, 10), Nint)
        else:
            s_pass2 = 1j * np.logspace(log(w1, 10), log(w2, 10), Nint)
        s_pass = np.sort(np.hstack((s_pass1, s_pass2)))
        Nint = 2 * Nint

        oldT0 = []
        EE = np.zeros((Nc, s_pass.shape[0]), dtype=complex)
        for k in range(s_pass.shape[0]):
            Y = fitcalcABCDE(s_pass[k], np.diag(A), B, C, D, np.zeros(Nc))
            G = Y.real
            if not colinterch:
                EE[:, k], EE_temp = LA.eig(G)
            else:
                DD, T0 = LA.eig(G)
                T0 = rot(T0.astype(complex)) # Minimizing phase angle of eigenvectors in least squares sense
                T0, DD = intercheig(T0, oldT0, np.diag(DD).copy(), Nc, k)
                oldT0 = T0.copy()
                EE[:, k] = np.diag(DD)

        # Identifying violations, picking minima for s2:
        s_pass_ind = np.full((1, s_pass.shape[0]), False)
        for row in range(Nc):
            if EE[row, 0] < 0:
                s_pass_ind[0, 0] = True
        for k in range(1, s_pass.shape[0] - 1):
            for row in range(Nc):
                if EE[row, k] < 0: # Violation
                    if EE[row, k] < EE[row, k - 1] and EE[row, k] < EE[row, k + 1]:
                        s_pass_ind[0, k] = True
        indices = np.nonzero(s_pass_ind == True)[1]
        s = np.hstack((s, s_pass[indices]))
        dum = np.amin(EE, axis=0) # Minimum of each column (across ports)
        g_pass2 = np.amin(dum) # Minimum value of EE
        ind = np.where(dum == g_pass2)
        smin2 = s_pass[ind] # Minimum eigenvalue is at smin2
        g_pass = np.amin(np.hstack((g_pass, g_pass2)))
        if g_pass == g_pass2:
            ind = 1
        else:
            ind = np.where(np.hstack((g_pass, g_pass2)) == g_pass)
        dums = np.hstack((smin, smin2))
        smin = dums[ind]

        g_pass = np.amin(np.hstack((g_pass2, np.amin(EE))))

    s_pass = s

    return s_pass, g_pass, smin