""" pr2ss.py

Author: Jennifer Houle
Date: 3/19/2020

This program is based off pr2ss.m from [4]. From [4],

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


def pr2ss(SER):
    """
    This function completes the state space model
    :param SER: R, poles
    :return: SER, now with A, B, C also filled in
    """
    R = SER['R']
    poles = SER['poles'].reshape(-1, 1)
    Nc = SER['D'].shape[0]
    N = R.shape[2]
    C = np.zeros((Nc, Nc * N), dtype=np.complex)
    A = np.zeros((Nc * N, 1), dtype=np.complex)
    B = np.zeros((Nc * N, Nc), dtype=np.complex)

    for m in range(N):
        Rdum = R[:, :, m]
        for n in range(Nc):
            ind = n * N + m
            C[:, ind] = Rdum[:, n].copy()
    for n in range(Nc):
        A[n * N: (n + 1) * N, 0] = poles[:, 0].copy()
        B[n * N: (n + 1) * N, n] = np.ones((N, 1), dtype=complex)[:, 0]
    A = np.diagflat(A)

    SER['A'] = A.copy()
    SER['B'] = B.copy()
    SER['C'] = C.copy()

    return SER