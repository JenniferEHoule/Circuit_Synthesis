""" fitcalc.py

Author: Jennifer Houle
Date: 3/27/2020

This program is based off fitcalcABCDE.m and fitcalcPRE from [4]. The purpose is to calculate Yfit
from the state space model as in Eq. (5) in [7]

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

[7] A. Semlyen, B. Gustavsen, "A Half-Size Singularity Test Matrix for Fast and Reliable
    Passivity Assessment of Rational Models," IEEE Trans. Power Delivery, vol. 24, no. 1,
    pp. 345-351, Jan. 2009.
"""


import numpy as np

def fitcalcABCDE(sk, A, B, C, D, E):
    """
    Calculate Yfit from the state space model as in Eq. (5) in [7]
    :param sk: frequency
    :param A: poles
    :param B: vector of 1's
    :param C: residues
    :param D: D values
    :param E: E values
    :return: Yfit calculation using the state space model
    """
    sk = sk
    Nc = D.shape[0]

    dum = np.tile(1 / (sk - A), (Nc, 1)).swapaxes(0, 1)
    C = C * dum.T
    Yfit = (C @ B) + D + (sk * E)
    return Yfit

def fitcalcPRE(s, SERA, SERC, SERD, SERE):
    """
    Calculate Yfit based on the state space model, summing SERC/SERA relationship
    across all poles to create a matrix with one value per Y param
    """
    Ns = s.shape[0]
    Nc = SERD.shape[0]
    N = SERA.shape[0]
    Yfit = np.zeros((Nc, Nc, Ns), dtype=complex)
    Y = np.zeros((Nc, Nc), dtype=complex)

    for k in range(Ns):
        tell = 0
        for row in range(Nc):
            for col in range(Nc):
                Y[row, col] = (SERD[row, col] + s[k] * SERE[row, col])[0]
                Y[row, col] = Y[row, col] + np.sum(SERC[row, col, :N] / (s[k] - SERA[:N]).T)
                tell = tell + 1
        Yfit[:Nc, :Nc, k] = Y
    return Yfit
