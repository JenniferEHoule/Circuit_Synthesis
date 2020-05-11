""" rot.py

Author: Jennifer Houle
Date: 3/25/2020

This program is based off rot.m from [4].

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

from math import atan2, cos, sin, pi

def rot(S):
    Nc = S.shape[0]
    SA = np.zeros((Nc, Nc), dtype=complex)
    SB = SA.copy()
    scale1 = np.zeros((Nc), dtype=complex)
    scale2 = scale1.copy()
    scale = scale1.copy()
    err1 = scale1.copy()
    err2 = scale1.copy()

    numerator = np.zeros(Nc)
    denominator = np.zeros(Nc)
    ang = np.zeros(Nc)

    # E-vectors
    for col in range(Nc):
        # Calculate the min/max value for square sum (error) of imaginary parts
        numerator[col] = 0.0
        denominator[col] = 0.0
        for j in range(Nc):
            numerator[col] = numerator[col] + (S[j, col]).imag * (S[j, col]).real       # Q2[j, i] * Q1[j, i]
            denominator[col] = denominator[col] + ((S[j, col]).real) ** 2 - ((S[j, col]).imag) ** 2 # Q1[j, i] ^2 - Q2[j, i] ^2
        numerator[col] = -2 * numerator[col]
        ang[col] = 0.5 * atan2(numerator[col], denominator[col])

        scale1[col] = cos(ang[col]) + 1j * sin(ang[col])
        scale2[col] = cos(ang[col] + pi / 2) + 1j * sin(ang[col] + pi / 2)

        # Deciding which solution (1,2) will produce the smallest error:
        for j in range(Nc):
            SA[j, col] = S[j, col] * scale1[col]
            SB[j, col] = S[j, col] * scale2[col]

        # Square sum (error) of solution:
        aaa = 0.0
        bbb = 0.0
        ccc = 0.0
        for j in range(Nc):
            aaa = aaa + (SA[j, col].imag) ** 2 # Q2A[j, i] ^2
            bbb = bbb + (SA[j, col]).real * (SA[j, col]).imag   # Q1A[j, i] * Q2A[j, i]
            ccc = ccc + ((SA[j, col]).real) ** 2    # Q1A[j, i] ^ 2
        err1[col] = aaa * cos(ang[col]) ** 2 + bbb * sin(2 * ang[col]) + ccc * sin(ang[col]) ** 2

        # Square sum (error) of solution #2
        aaa = 0.0
        bbb = 0.0
        ccc = 0.0
        for j in range(Nc):
            aaa = aaa + (SB[j, col].imag) ** 2 # Q2A[j, i] ^2
            bbb = bbb + (SB[j, col]).real * (SB[j, col]).imag   # Q1A[j, i] * Q2A[j, i]
            ccc = ccc + ((SB[j, col]).real) ** 2    # Q1A[j, i] ^ 2
        err2[col] = aaa * cos(ang[col]) ** 2 + bbb * sin(2 * ang[col]) + ccc * sin(ang[col]) ** 2

        # Picking the solution (1,2) with the smallest square sum:
        if(err1[col] < err2[col]):
            scale[col] = scale1[col]
        else:
            scale[col] = scale2[col]

        # Rotating e-vector
        S[:, col] = S[:, col] * scale[col]

        # Rotating e-vector
        S[:, col] = S[:, col] * scale[col]

    return S