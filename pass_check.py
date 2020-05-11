""" pass_check.py

Author: Jennifer Houle
Date: 3/27/2020

This program is based off RPdriver.m functions from [4]. This will identify violating
intervals based off singularities of the eigenvalues of Eq. (23a) in [7]. It returns the
intervals in a matrix, with each column indicating the beginning and the end of each
violating interva.

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
import scipy.linalg as LA

from utils import chop, PoleTypes, find_which_poles_are_complex
from fitcalc import fitcalcABCDE

def pass_check_Y(A, C, D):
    """
    Input state-space model with diagonal A (poles) with complex conjugate formulation and return
    a matrix with 2 rows, and each column denotes the start and end frequency of non-passive band
    A violation extending to infinity is denoted by s_end=j*1e16

    :param A: poles in a vector (num_poles)
    :param C: residues in a 3D matrix (ports x ports x num_poles)
    :param D: D values in a 2D matrix (ports x ports)
    :return:
    """

    # This will rearrange A, B, C
    Nc = D.shape[0]
    N = A.shape[0]
    tell = 0
    CC = np.zeros((Nc, Nc * N), dtype=complex)
    B = np.ones((N, 1))

    for col in range(Nc):
        if col == 0:
            AA = np.diagflat(A)
            BB = B
        else:
            AA = LA.block_diag(AA, np.diagflat(A))
            BB = LA.block_diag(BB, B)
        for row in range(col, Nc):
            CC[row, col * N: (col + 1) * N] = C[row, col, :]
            CC[col, row * N: (row + 1) * N] = C[row, col, :]
    A = AA.copy()
    B = BB.copy()
    C = CC.copy()

    Acmplx = A.copy()
    Bcmplx = B.copy()
    Ccmplx = C.copy()
    Dcmplx = D.copy()

    A = chop(A)     # Get rid of tiny imaginary numbers
    TOL = 1e-5      # Use this instead of 0 to look for tiny sums

    # Convert to real-only configuration
    if np.sum(A - np.diagflat(np.diag(A))) < TOL:
        cindex, A, B, C = separate_real_imag_in_state_space(A, B, C)

    N = A.shape[0]
    Nc = D.shape[0]

    E = np.zeros((Nc, Nc))
    if np.sum(LA.eig(D) == 0):
        Ahat = LA.solve(A, np.eye(N))
        Bhat = - Ahat @ B
        Chat = C * Ahat
        Dhat = D - C * Ahat * B
        A = Ahat.copy()
        B = Bhat.copy()
        C = Chat.copy()
        D = Dhat.copy()
    S1 = A @ (B @ (np.linalg.matrix_power(D, -1)) @ C - A) # Eq. (23a) in [7]

    wS1, wS2 = LA.eig(S1)
    wS1 = np.sqrt(wS1)          # See note below Eq. (23a) in [17]
    if np.sum(LA.eig(Dcmplx) == 0) > 0:
        ws1 = 1 / wS1
    ind = np.nonzero(np.abs(wS1.imag) < 1e-6)
    wS1 = (wS1[ind]).real
    sing_w = np.sort(wS1)

    if sing_w.shape[0] == 0:
        intervals = np.array([])
        return intervals

    A = Acmplx.copy()
    B = Bcmplx.copy()
    C = Ccmplx.copy()
    D = Dcmplx.copy()

    # Establishing frequency list at midpoint of all bands defined by sing_w
    midw = np.zeros((1 + sing_w.shape[0], 1), dtype=complex)
    midw[0] = sing_w[0] / 2
    midw[-1] = 2 * sing_w[-1]
    for k in range(sing_w.shape[0] - 1):
        midw[k + 1] = (sing_w[k] + sing_w[k + 1]) / 2

    EE = np.zeros((Nc, midw.shape[0]), dtype=complex)
    viol = np.zeros(midw.shape[0])

    # Checking passivity at all midpoints
    for k in range(midw.shape[0]):
        sk = 1j * midw[k]
        G = (fitcalcABCDE(sk[0], np.diag(A), B, C, D, E)).real
        EE[:, k], EE_temp = LA.eig(G)
        if np.any(EE[:, k] < 0, axis=0):
            viol[k] = 1
        else:
            viol[k] = 0

    # Establishing intervals for passivity violations:
    intervals = np.empty((2,0))
    for k in range(midw.shape[0]):
        if viol[k] == 1:
            if k == 0:
                intervals = (np.vstack((0, sing_w[0])))
            elif k == midw.shape[0] - 1:
                intervals = np.hstack((intervals, (np.vstack((sing_w[k - 1], 1e16)))))
            else:
                intervals = np.hstack((intervals, (np.vstack((sing_w[k - 1], sing_w[k])))))

    if not np.any(intervals):
        wintervals = intervals.copy()
        return wintervals

    # Collapsing overlapping bands:
    killindex = 0
    for k in range(1, intervals.shape[1]):
        if intervals[1, k - 1] == intervals[0, k]: # An overlap exists
            intervals[1, k - 1] = intervals[1, k]
            intervals[:, k] = intervals[:, k - 1]
            killindex = np.append(killindex, k - 1)

    # Delete any intervals with killindex == 1
    if np.any(killindex) != 0:
        intervals = np.delete(intervals, killindex, axis=1)
    wintervals = intervals.copy()
    return wintervals


def separate_real_imag_in_state_space(A, B, C):
    """
    This separates the real and imaginary state space A, B, C and puts them in the cofiguration similar Eq. B.2 in [1].
    :param A: SER[A] - complex
    :param B: SER[B] - complex
    :param C: SER[C] - complex
    :return: A, B, C now with real and imaginary parts separated
    """
    N = A.shape[0]
    cindex = find_which_poles_are_complex(N, A)
    n = 0
    for m in range(N):
        if cindex[m] == PoleTypes.COMPLEX_FIRST:
            a_real, a_imag = divide_real_imag(A[n, n])
            c_real, c_imag = divide_real_imag(C[:, n])
            b = B[n, :].copy()
            b1 = 2 * b.real
            b2 = -2 * b.imag
            Ablock = np.array([[a_real, a_imag], [-a_imag, a_real]])
            A[n:n + 2, n:n + 2] = Ablock.copy()
            C[:, n] = c_real.copy()
            C[:, n + 1] = c_imag.copy()
            B[n, :] = b1.copy()
            B[n + 1, :] = b2.copy()
        n = n + 1
    return cindex, A, B, C


def divide_real_imag(number):
    """
    Retruns the complex number divided
    :param number: complex number
    :return: real part of number, imaginary part of number
    """
    return number.real.copy(), number.imag.copy()

