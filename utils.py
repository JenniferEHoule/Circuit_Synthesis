""" utils.py

Author: Jennifer Houle
Date: 3/19/2020

This program contains supporting functions and classes for VFdriver.py and RPdriver.py.

PoleTypes, AsympOptions, WeightParam, and OutputLevel are classes defining certain options for
VFdriver.py and/or RPdriver.py, based on [1].

chop, find_which_poles_are_complex, and sort_eigenvalues_eigenvectors_to_match_matlab are functions
used to help align MATLAB and Python results.

[1] B. Gustavsen, Matrix Fitting Toolbox, The Vector Fitting Website.
    March 20, 2013. Accessed on: Feb. 25, 2020. [Online]. Available:
    https://www.sintef.no/projectweb/vectorfitting/downloads/matrix-fitting-toolbox/.

"""

from enum import Enum, auto

import numpy as np


class PoleTypes(Enum):
    """
    This class contains the different pole types used for indexing matrices in VFIT3
    """
    REAL = auto()               # 0 in MATLAB
    COMPLEX_FIRST = auto()      # 1 in MATLAB
    COMPLEX_SECOND = auto()     # 2 in MATLAB


class AsympOptions(Enum):
    """
    This class contains the options for asymp. This can include D only, include D and E, or include neither D nor E
    See Eq. (4) in [2].
    """
    NONE = 'NONE'
    D = 'D'
    DE = 'DE'


def chop(arr):
    """
    Replaces approximate imaginary numbers in arr that are close to zero with exactly zero.
    This minimizes the chance a solution pole will be mistaken for complex when it should be rea.
    """
    near_zero_imag = np.isclose(arr.imag, 0, atol=1e-1)
    arr[near_zero_imag] = arr[near_zero_imag].real
    return arr

class WeightParam(Enum):
    """
    common_1    --> weight=1 for all elements in Least Sq. problem, at all freq.
    indiv_norm  --> weight(s)=1/abs(Hij(s))      ; indvidual element weight
    indiv_sqrt  --> weight(s)=1/sqrt(abs(Hij(s))); indvidual element weight
    common_norm --> weight(s)=1/norm(H(s))       ; common weight for all matrix elements
    common_sqrt --> weight(s)=1/sqrt(norm(H(s))  ; common weight for all matrix elements
    """
    common_1 = 'common_1'       # 1 in MATLAB
    indiv_norm = 'indiv_norm'   # 2 in MATLAB
    indiv_sqrt = 'indiv_sqrt '  # 3 in MATLAB
    common_norm = 'common_norm' # 4 in MATLAB
    common_sqrt = 'common_sqrt' # 5 in MATLAB

def find_which_poles_are_complex(N, LAMBD):
    """
    :param LAMBD:  Diagonal matrix of the initial guess for the poles. Complex conjugate pairs must be together.
    :return: cindex - list of the types of poles (real, complex_first, complex_second) for LAMBD
    """
    cindex = [PoleTypes.REAL for _ in range(N)]
    for m in range(0, N - 1):
        if LAMBD.imag[m, m]:
            if m == 0:
                cindex[m] = PoleTypes.COMPLEX_FIRST
            else:
                if cindex[m - 1] == PoleTypes.REAL or cindex[m - 1] == PoleTypes.COMPLEX_SECOND:
                    cindex[m] = PoleTypes.COMPLEX_FIRST
                    cindex[m + 1] = PoleTypes.COMPLEX_SECOND
                else:
                    cindex[m] = PoleTypes.COMPLEX_SECOND
    return cindex

def sort_eigenvalues_eigenvectors_to_match_matlab(eigenvalues, eigenvectors):
    idx = eigenvalues.argsort()[::1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

class OutputLevel(Enum):
    """
    This class contains the options for ouptput level.
    'min' : Minimal level of output information to screen
    'max' : Maximum level of output information to screen
    """
    min = 'min' # 0 in MATLAB
    max = 'max' # 1 in MATLAB
