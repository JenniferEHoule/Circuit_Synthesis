""" intercheig.py

Author: Jennifer Houle
Date: 3/25/2020

This program is based off intercheig.m from [4].

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

def intercheig(V, oldV, D, Nc, fstep):
    """
    Sort eigenvalues, eigenvectors
    """
    if fstep > 1:
        UGH = np.abs((oldV.conj().T @ V))
        dot = np.zeros(Nc)
        ind = dot.copy()
        taken = [False] * Nc

        for ii in range(Nc):
            ilargest = 0
            rlargest = 0
            for j in range(Nc):
                dotprod = UGH[ii, j].copy()
                if dotprod > rlargest:
                    rlargest = np.abs(dotprod.real)
                    ilargest = j
            dot[ii] = rlargest
            ind[ii] = ii
            taken[ii] = False

        # Sorting inner products abs(realde1) in descending order:
        for ii in range(Nc):
            for j in range(Nc - 1):
                if(dot[j] < dot[j + 1]):
                    hjelp = dot[j + 1]
                    ihjelp = ind[j + 1]
                    dot[j + 1] = dot[j]
                    ind[j + 1] = ind[j]
                    dot[j] = hjelp
                    ind[j] = ihjelp

        # Doing the interchange in a prioritized sequence:
        for l in range(Nc):
            ii = int(ind[l])
            ilargest = 0
            rlargest = 0

            for j in range(Nc):
                if not taken[j]:
                    dotprod = UGH[ii, j].copy()
                    if dotprod > rlargest:
                        rlargest = np.abs(dotprod.real)
                        ilargest = j

            taken[ii] = True

            hjelp = V[:, ii].copy()
            V[:, ii] = V[:, ilargest].copy()
            V[:, ilargest] = hjelp

            hjelp = D[ii, ii].copy()
            D[ii, ii] = D[ilargest, ilargest].copy()
            D[ilargest, ilargest] = hjelp

            dum = UGH[:, ii].copy()
            UGH[:, ii] = UGH[:, ilargest].copy()
            UGH[:, ilargest] = dum

        # Finding out whether the direction of e-vectors are 180 deg is done by comparing
        # sign of dotproducts of e-vectors, for new and old V-matrix
        for ii in range(Nc):
            dotprod = oldV[:, ii].conj().T @ V[:, j]
            if np.sign(dotprod.real) < 0:
                V[:, ii] = -V[:, ii]

    return V, D