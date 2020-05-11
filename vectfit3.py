""" vectfit3.py
This is based directly off the program vectfit3.m (details below author information)
This implements the VFIT 3 algorithm.

Date: 1/25/2020
Author: Jennifer Houle

B. Gustavsen, VFIT3, The Vector Fitting Website. March 20, 2013. Accessed on:
Feb. 22, 2020. [Online]. Available: https://www.sintef.no/projectweb/vectfit/downloads/vfut3/.

APPROACH:
The identification is done using the pole relocating method known as Vector Fitting [1],
with relaxed non-triviality constraint for faster convergence and smaller fitting errors [2],
and utilization of matrix structure for fast solution of the pole identification step [3].

********************************************************************************
NOTE: The use of this program is limited to NON-COMMERCIAL usage only.
If the program code (or a modified version) is used in a scientific work,
then reference should be made to the following:

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
********************************************************************************
This example script is part of the vector fitting package (VFIT3.zip)
Last revised: 08.08.2008.
Created by:   Bjorn Gustavsen.
"""

from math import pi, sqrt

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

from utils import PoleTypes, AsympOptions, chop


class VectorFit3:
    """
    This implements the Vector Fitting algorithm
    """
    DEFAULT_OPTIONS = dict(
        stable=True,
        asymp=AsympOptions.D,
        skip_pole=False,
        skip_res=False,
        cmplx_ss=True,
        spy1=False,
        spy2=True,
        logx=True,
        logy=True,
        errplot=True,
        phaseplot=False,
        legend=True
    )

    def __init__(self, **options):
        """
        Sets up the options by merging the default options with any the user selects.
        User selected options have priority
        """
        if 'asymp' in options and not isinstance(options['asymp'], AsympOptions):
            options['asymp'] = AsympOptions(options['asymp'])
        self.options = {**self.DEFAULT_OPTIONS, **options}

    def find_which_poles_are_complex(self, LAMBD):
        """
        :param LAMBD:  Diagonal matrix of the initial guess for the poles. Complex conjugate pairs must be together.
        :return: cindex - list of the types of poles (real, complex_first, complex_second) for LAMBD
        """
        cindex = [PoleTypes.REAL for _ in range(self.N)]
        for m in range(0, self.N - 1):
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

    def do_fit(self, f, s, poles, weight):
        """
        This is the main method for vector fitting, implementing [1], [2], and [3]

        :param f: Data for the frequency response. Number_of_ports x number_of_data_points
        :param s: Frequency data corresponding to f. 1D array with length number_of_data_points
        :param poles: Initial pole guesses. 1D array with length number_of_poles. Complex conjugate pairs
        should be adjacent.
        :param weight: Weighting is optional; otherwise this should be set to 1's. Array of size
        number_of_ports x number_of_data_points
        :return:
            SER: State-space model. Dictionary containing A, B, C, D, and E (as applicable from input options).
            See Eq. (1) and (5) in [2]
            poles: Updated pole locations. 1D array with length number_of_poles.
            rmserr: Error with the final fit. Floating point number
        """

        if len(f.shape) == 1:
            # This converts an array of dimensions (points,) to (points,1) to allow matrix math
            f = f.reshape(1, -1)

        self.f = f
        self.s = s
        self.poles = poles
        self.weight = weight

        self.N = self.poles.shape[1] # Number of poles
        self.Ns = self.s.shape[0]    # Number of data points for frequency and response
        self.Nc = self.f.shape[0]    # Number of ports

        if self.s.shape[0] < 1:
            self.s = self.s.T.copy()    # Ensure the shape of s is (number_of_data_points,)

        self.check_input_array_dimensions()

        LAMBD = np.diag(self.poles[0])  # Convert the poles to a diagonal matrix
        SERA = self.poles.copy()               # Initial SERA is the same as the input poles (NOT diagonal matrix)
        roetter = poles.copy()                 # Roetter is also set to the input poles (NOT diagonal matrix)
        fit = np.zeros((self.Nc, self.Ns))  # fit is set up; this will be the data generated with the newly calculated
                                            # poles and residues

        # Manipulate weight to be the correct shape and determine if weight is common across all ports
        self.weight = np.transpose(self.weight)
        if self.weight.shape[1] == 1:
            temp_weight = self.weight
            for ind in range(self.N - 1):
                temp_weight = np.hstack((temp_weight, self.weight))
            common_weight = 1
            self.weight = temp_weight
        elif weight.shape[0] == self.Nc:
            common_weight = 0
        else:
            raise ValueError("Invalid size of array weight")

        # Pole Identification
        if not self.options['skip_pole']:
            SERA, roetter = self.find_pole(LAMBD, common_weight)

        # Residue Identification:
        if not self.options['skip_res']:
            B, C, D, E, rmserr, fit = self.find_residue(roetter, common_weight)
        else:
            B = np.ones((self.N, 1))
            C = np.zeros((self.Nc, self.N), dtype=complex)
            D = np.zeros((self.Nc, self.Nc), dtype=complex)
            E = np.zeros((self.Nc, self.Nc), dtype=complex)
            rmserr = 0
        A = SERA.copy()
        self.poles = SERA.reshape(1, -1).copy()
        # Convert into real state-space model
        SER = self.convert_to_real_state_space_model(A, B, C, D, E)

        return SER, self.poles, rmserr, fit

    def convert_to_real_state_space_model(self, A, B, C, D, E):
        if not self.options['cmplx_ss']:
            A = np.diag(A)
            cindex = self.find_which_poles_are_complex(A)

            n = 0
            for m in range(self.N):
                n = n + 1
                if cindex[m] == PoleTypes.REAL:
                    a = A[n, n]
                    a1 = a.real
                    a2 = a.imag
                    c = C[:, n]
                    c1 = c.real
                    c2 = c.imag
                    b = B[n, :]
                    b1 = 2 * b.real
                    b2 = -2 * b.imag
                    Ablock = np.array((a1, a2), (-a2, a1))

                    A[n:n + 1, n: n + 1] = Ablock
                    C[:, n] = c1
                    C[:, n + 1] = c2
                    B[n, :] = b1
                    B[n + 1, :] = b2
        else:
            A = np.diag(A)
        SER = dict(
            A=A,
            B=B,
            C=C,
            D=D,
            E=E
        )
        return SER

    def build_system_matrix(self, cindex, LAMBD, weight=1):
        """
        This creates the Dk matrix, which is then used to create A in Eq. (A.3) in [1].
        This is the 1/(sk-a1) ... 1/(sk-aN) part of that equation, weighted.
        This function is identical for each pole, since the 'f' data is not included here.
        :param cindex: List of the types of poles (real, complex_first, complex_second) for LAMBD
        :param LAMBD: Diagonal matrix of the initial poles. Complex conjugate pairs must be together.
        :param weight: Array of size number_of_ports x number_of_data_points to give weighting.
        This is set to 1 if no input is given.
        :return: Dk - matrix of data used to create A in Eq. (A.3) in [1].
        Dimensions are (number_of_data_points x (number_of_poles)
        """
        Dk = np.zeros((self.Ns, self.N), dtype=np.complex)
        for m in range(0, self.N):
            if cindex[m] == PoleTypes.REAL:
                Dk[:, m] = weight / (self.s - LAMBD[m, m])
            elif cindex[m] == PoleTypes.COMPLEX_FIRST:
                Dk[:, m] = weight / (self.s - LAMBD[m, m]) + weight / (self.s - np.conj(LAMBD[m, m]))
                Dk[:, m + 1] = weight * 1j / (self.s - LAMBD[m, m]) - weight * 1j / (self.s - np.conj(LAMBD[m, m]))
        return Dk

    def check_input_array_dimensions(self):
        """
        This causes errors to be output if the input s, f, and weight are incompatible dimensions.
        :return:
        """
        if self.s.shape[0] != self.f.shape[1]:
            raise ValueError(f"Dimension of s ({self.s.shape[0]}) is not equal to dimension of f ({self.f.shape[1]})")
        if self.s.shape[0] != self.weight.shape[1]:
            raise ValueError(
                f"Dimension of s ({self.s.shape[0]}) is not equal to dimension of weight ({self.weight.shape[1]})")
        if self.weight.shape[1] != self.f.shape[1]:
            raise ValueError(
                f"Dimension of weight ({self.weight.shape[1]}) is not equal to dimension of f ({self.f.shape[1]})")
        if self.poles.shape[1] >= self.f.shape[1] - 1:
            raise ValueError(
                f"Dimension of poles ({self.poles.shape[1]}) is too big compared to dimension of f ({self.f.shape[1]})")

    def set_scale_legend_options(self):
        """
        This sets the logarithmic scaling and legend options.
        :return:
        """
        if self.options['logx']:
            if self.options['logy']:
                plt.xscale('log')
                plt.yscale('log')
            else:
                plt.xscale('log')
        else:
            if self.options['logy']:
                plt.yscale('log')
        if self.options['legend']:
            plt.legend()

    def calculate_x(self, AA, bb, Escale):
        """
        Calculate x, which is the least squares solution to AA*x=bb, Eq. (6), Eq. (A.8) in [1]
        :param AA: Matrix corresponding to Eq. (A.3), (A.6) in [1].
        :param bb: Vector corresponding to Eq. (A.4), (A.8) in [1].
        :param Escale: Used for normalization of AA in [1].
        :return: x: Least squares solution corresponding to Eq. (6), Eq. (A.8) in [1].
        """
        for col in range(AA.shape[1]):
            # Normalize AA
            Escale[col, 0] = 1 / np.linalg.norm(AA[:, col])
            AA[:, col] = Escale[col, 0] * AA[:, col]

        x, *other_stuff = np.linalg.lstsq(AA, bb, rcond=None)  # We only care about x as an output
        x = x * Escale # Again apply the normalization.
        return x

    def find_pole(self, LAMBD, common_weight):
        """
        This function is used to find the poles of f by finding the zeros of sigma using the method described in [3].
        :param LAMBD: Diagonal matrix of the initial poles.
        :param common_weight: Determines whether there is a single weighting used for all ports.
        :return:
            roetter: The zeros of the sigma function (the new poles of f).
            SERA: Transpose of roetter
        """

        # Set up an offset so make room for potential D and E coefficients in Eq. (1) in [2].
        if self.options['asymp'] == AsympOptions.NONE:
            offs = 0
        elif self.options['asymp'] == AsympOptions.D:
            offs = 1
        else:
            offs = 2

        # Finding out which starting poles are complex
        cindex = self.find_which_poles_are_complex(LAMBD)

        # Building system matrix
        Dk = self.build_system_matrix(cindex, LAMBD)

        # Use the Dk matrix to create something closer to Eq. (A.1) in [1] for each port entry.
        if self.options['asymp'] in (AsympOptions.NONE, AsympOptions.D):
            # Add a column of 1's after the Dk data
            Dk = np.column_stack((Dk, np.ones(self.Ns))).copy()
        elif self.options['asymp'] == AsympOptions.DE:
            # Add a column of 1's after the Dk data and a column of the s data as in Eq. (A.3) in [1]
            Dk = np.column_stack((Dk, np.ones(self.Ns), self.s)).copy()

        # Scaling for last row of LS-problem (pole identification). Calculate Eq. (9) in [2]
        scale = 0
        for m in range(0, self.Nc):
            if self.weight.shape[1] == 1:
                scale = scale + (np.linalg.norm(self.weight[:, 0] * np.transpose(self.f[m]))) ** 2
            else:
                scale = scale + (np.linalg.norm(self.weight[:, m] * np.transpose(self.f[m]))) ** 2
        scale = sqrt(scale) / self.Ns

        # Use relaxed nontriviality constraint (removed a non-relaxed option entirely)
        AA = np.zeros((self.Nc * (self.N + 1), self.N + 1))  # Set up AA to be used in Eq. (6), (A.8) in [1]
        bb = np.zeros((self.Nc * (self.N + 1), 1))           # Set up bb to be used in Eq. (6), (A.8) in [1]
        Escale = np.zeros((self.N + 1, 1))

        for n in range(self.Nc):
            A = np.zeros((self.Ns, (self.N + offs) + self.N + 1), dtype=np.complex)
            # offs was set earlier based on the D, DE, or neither D nor E options

            if common_weight == 1:
                weig = self.weight[:, 0]
            else:
                weig = self.weight[:, n]

            for m in range(self.N + offs):  # Left Block in Eq. (10) in [3]
                A[0:self.Ns, m] = weig * Dk[0:self.Ns, m]

            inda = self.N + offs            # Dk already contains '1' and 's' data if applicable for D/E opts.
            for m in range(self.N + 1):     # Right block in Eq. (10) in [3]
                A[0:self.Ns, inda + m] = -weig * Dk[0:self.Ns, m] * self.f[n, 0:self.Ns].T

            A = np.vstack((np.real(A), np.imag(A))) # Stack the real and imaginary components as in Eq. (7) in [3]

            # Integral criterion for sigma (this takes care of Eq. (8) in [2]
            offset = self.N + offs
            if n + 1 == self.Nc:
                A_temp = np.zeros((1, A.shape[1]))
                for mm in range(self.N + 1):
                    A_temp[0, offset + mm] = np.real(scale * np.sum(Dk[:, mm]))
                # A_temp is a vector  accounting for Eq. (8) in [2]; it is stacked below the last row of A
                # A is called [X -HvX] in Eq. (10) in [3].  Used to avoid the null solution by adding to the LS problem.
                A = np.vstack((A, A_temp))

            Q, R = np.linalg.qr(A)  # Solve as in Eq. (10) in [3] to implement the fast implementation of VF
            ind1 = self.N + offs
            ind2 = self.N + offs + self.N + 1
            R22 = R[ind1:ind2, ind1:ind2]   # R22 is used in Eq. (11) in [3]
            AA[(n) * (self.N + 1):(n + 1) * (self.N + 1), :] = R22.copy()
                                                                # Fill in the vector of R22's for Eq. (11) in 3

            if n + 1 == self.Nc:
                bb[((n) * (self.N + 1)):(n + 1) * (self.N + 1), 0] = Q[-1, self.N + offs:].conj().T * self.Ns * scale
                # This covers the right side of Eq. (11) in [3], with scale from above taking care of Hv weighted
                # with Eq. (8), (9) in [2].

        x = self.calculate_x(AA, bb, Escale)  # Calculate the C tilda residues in Eq. (11) in [3]

        C = x[:-1].copy().astype(np.complex)  # The C tilda residues become C sigma in Eq. (9) in [3]
        D = x[-1].reshape(1, -1)              # The last entry in the x vector becomes D in Eq. (9) in [3]

        # We now change back to make C complex
        for m in range(self.N):
            if cindex[m] == PoleTypes.COMPLEX_FIRST:
                r1 = C[m].copy()
                r2 = C[m + 1].copy()
                C[m] = r1 + 1j * r2
                C[m + 1] = r1 - 1j * r2

        if self.options['spy1']:
            self.plot_sigma(LAMBD, C, D)

        # We now calculate the zeros for sigma
        m = -1
        B = np.ones((LAMBD.shape[0], 1))    # B is set to a default vector of 1's

        # LAMBD is calculated using the A hat for complex numbers in Eq. (B.2) in [1],
        # which is A sigma in Eq. (9) in [3]
        # B is changed to [[2],[0]] according to Eq. (B.2) in [1] for complex numbers
        # c tilda prime is also adjustded for complex numbers according to Eq. (B.2) in [1]
        for n in range(self.N):
            m = m + 1
            if m < self.N - 1:
                if LAMBD.imag[m, m]:
                    LAMBD[m + 1, m] = -np.imag(LAMBD[m, m])
                    LAMBD[m, m + 1] = np.imag(LAMBD[m, m])
                    LAMBD[m, m] = np.real(LAMBD[m, m])
                    LAMBD[m + 1, m + 1] = LAMBD[m, m].copy()
                    B[m, 0] = 2
                    B[m + 1, 0] = 0
                    koko = C[m, 0]
                    C[m] = koko.real
                    C[m + 1] = koko.imag
                    m = m + 1

        ZER = LAMBD - B * C.T / D[0, 0]  # Eq. (9) in [3]
        roetter = LA.eigvals(ZER)   # The rest of Eq. (9) in [3] to find the zeros of sigma / new poles (called roetter)
        roetter = chop(roetter)     # Get rid of tiny imaginary numbers so the program doesn't think reals are complex

        unstables = np.real(roetter) > 0
        if self.options['stable']:
            # Forcing unstable poles to be stable...
            roetter[unstables] = roetter[unstables] - 2 * np.real(roetter[unstables])
        self.N = roetter.shape[0]   # Make sure N matches the new number of poles.

        roetter = roetter[np.argsort(np.abs(roetter))]  # Magnitude sort
        roetter = roetter[np.argsort(roetter.imag != 0, kind='stable')]  # Move reals to the beginning
        roetter = roetter - 2 * 1j * roetter.imag   # Force the sorting to match that of MATLAB (negative imaginary
                                                    # then positive imaginary for complex conjugate pairs).
        SERA = roetter.T  # Both of these are the new poles

        return SERA, roetter

    def find_residue(self, roetter, common_weight):
        """
        This finds the residues with the new poles
        :param roetter: Vector of the new poles
        :param common_weight: Parameter that is '1' if all ports use the same weight vector
        :return: The components of Eq. (1) in [2] and RMS error
            B: Vector of 1's
            C: Matrix containing the residues for each port's data
            D: Vector of the d values for each port's data
            E: Vector of the e values for each port's data
            rmserr: RMS error (generated fit compared with original f data)
        """
        LAMBD = np.diag(roetter)
        cindex = self.find_which_poles_are_complex(LAMBD)

        SERD = 0
        SERE = 0

        # Initialize A, BB to the proper size to be used in Eq. (6) in [1]
        if self.options['asymp'] == AsympOptions.NONE:
            A = np.zeros((2 * self.Ns, self.N), dtype=complex)
            BB = np.zeros((2 * self.Ns, self.Nc), dtype=complex)
        elif self.options['asymp'] == AsympOptions.D:
            A = np.zeros((2 * self.Ns, self.N + 1), dtype=complex)
            BB = np.zeros((2 * self.Ns, self.Nc), dtype=complex)
        else:
            A = np.zeros((2 * self.Ns, self.N + 2), dtype=complex)
            BB = np.zeros((2 * self.Ns, self.Nc), dtype=complex)

        # This option is used if all port data has the same weight.  The workings are very similar to that described
        # in the find_new_poles method
        if common_weight == 1:
            Dk = self.build_system_matrix(cindex, LAMBD, self.weight[:, 0]) # All data has same weight; only one
                                                                            # column of data is needed

            SERD = np.zeros((self.Nc, 1), dtype=np.complex)
            SERE = np.zeros((self.Nc, 1), dtype=np.complex)

            A, BB = self.initialize_A_BB(A, BB, Dk)

            # Clear Escale
            Escale = np.zeros((1, A.shape[1]))

            for col in range(A.shape[1]):
                Escale[0, col] = np.linalg.norm(A[:, col], ord=2)
                                    # The ord=2 doesn't seem to matter but I'll leave it for consistency
                A[:, col] = A[:, col] / Escale[0, col]

            X, *_ = np.linalg.lstsq(A, BB, rcond=None)  # This calculates the residues using Eq. (6) in [1]

            for n in range(0, self.Nc):
                X[:, n] = X[:, n] / Escale[0]           # Normalization

            X = X.T
            C = X[:, 0:self.N]  # Contains the residues c1 to CN in Eq. (A.4) in [1]

            # If applicable, D and E are pulled out as shown in Eq. (4) in [2]
            if self.options['asymp'] == AsympOptions.D:
                SERD = X[:, self.N]
            elif self.options['asymp'] == AsympOptions.DE:
                SERE = X[:, self.N + 1]
                SERD = X[:, self.N]

        else:  # if common weight != 1
            # Building system matrix
            Dk = self.build_system_matrix(cindex, LAMBD, 1)  # The 1 is the weight, so this is evenly weighted here

            SERD = np.zeros((self.Nc, 1), dtype=np.complex)
            SERE = np.zeros((self.Nc, 1), dtype=np.complex)
            C = np.zeros((self.Nc, self.N), dtype=np.complex)

            # This is all set up very similarly to the find_new_poles method. The difference is the weighting is
            # different for each set of port data
            for n in range(0, self.Nc):
                if self.options['asymp'] == AsympOptions.NONE:
                    A[0:self.Ns, 0:self.N] = Dk.copy()
                elif self.options['asymp'] == AsympOptions.D:
                    A[0:self.Ns, 0:self.N] = Dk.copy()
                    A[0:self.Ns, self.N] = 1
                else:
                    A[0:self.Ns, 0:self.N] = Dk.copy()
                    A[0:self.Ns, self.N] = 1
                    A[0:self.Ns, self.N + 1] = self.s.copy()

                for m in range(A.shape[1]):
                    A[0:self.Ns, m] = self.weight[:, n] * A[0:self.Ns, m]

                BB = self.weight[:, n] * self.f[n, :].T
                A[self.Ns:2 * self.Ns, :] = A[0:self.Ns, :].imag
                A[0:self.Ns, :] = A[0:self.Ns, :].real
                BB = np.hstack((BB.real, BB.imag)).reshape(-1,1)

                if self.options['asymp'] == AsympOptions.D:
                    A[0:self.Ns, self.N] = A[0:self.Ns, self.N].copy()
                elif self.options['asymp'] == AsympOptions.DE:
                    A[0:self.Ns, self.N] = A[0:self.Ns, self.N].copy()
                    A[self.Ns:2 * self.Ns, self.N + 1] = A[self.Ns:2 * self.Ns, self.N + 1].copy()

                # Clear Escale
                Escale = np.zeros((1, A.shape[1]))

                for col in range(A.shape[1]):
                    Escale[0, col] = np.linalg.norm(A[:, col], ord=2)
                                            # The ord=2 doesn't seem to matter but I'll leave it for consistency
                    A[:, col] = A[:, col] / Escale[0, col]

                x, *_ = np.linalg.lstsq(A, BB, rcond=None)  # This calculates the residues using Eq. (6) in [1]

                x[:, 0] = x[:, 0] / Escale[0]               # Normalization

                x = x.T
                C[n, 0:self.N] = x[:, 0:self.N].copy()

                # If applicable, D and E are pulled out as shown in Eq. (4) in [2]
                if self.options['asymp'] == AsympOptions.D:
                    SERD[n] = x[0, self.N].copy()
                elif self.options['asymp'] == AsympOptions.DE:
                    SERE[n] = x[0, self.N + 1].copy()
                    SERD[n] = x[0, self.N].copy()

        # We now change back to make C complex
        for m in range(0, self.N):
            if cindex[m] == PoleTypes.COMPLEX_FIRST:
                for n in range(0, self.Nc):
                    r1 = C[n, m]
                    r2 = C[n, m + 1]
                    C[n, m] = r1 + 1j * r2
                    C[n, m + 1] = r1 - 1j * r2

        B = np.ones((self.N, 1))
        SERA = roetter
        SERB = B
        SERC = C

        # Use the values calculated to plug into Eq. (4) in [2] to form a 'fit' for comparison
        fit = np.zeros((self.Nc, self.Ns), dtype=np.complex)
        Dk = np.zeros((self.Ns, self.N), dtype=np.complex)
        for m in range(0, self.N):
            Dk[:, m] = 1 / (self.s - SERA[m])
        for n in range(0, self.Nc):
            fit[n, :] = Dk.dot(SERC[n, :])
            if self.options['asymp'] == AsympOptions.D:
                fit[n, :] = fit[n, :] + SERD[n]
            elif self.options['asymp'] == AsympOptions.DE:
                fit[n, :] = fit[n, :] + SERD[n] + self.s.T * SERE[n]

        fit = fit.T
        f = self.f.T
        diff = fit - f
        rmserr = np.sqrt(np.sum(np.abs(diff ** 2))) / np.sqrt(self.Nc * self.Ns)    # Calculate the RMS error

        if self.options['spy2']:
            self.plot_magnitude_and_phase(fit)

        B = SERB
        C = SERC
        D = SERD
        E = SERE

        return B, C, D, E, rmserr, fit

    def initialize_A_BB(self, A, BB, Dk):
        """
        This sets up the A and B Eq. (6) in [1]
        :param A: Matrix to hold the A data (see Eq. (A.3), (A.8) in [1])
        :param BB: Vector to hold the B data (see Eq. (A.4), (A.8) in [1])
        :param Dk: Matrix containing 1/(s-ai) data as shown in Eq. (A.3) in [1]
        :return: A, BB set up with Dk / weight to match Eq. (A.3), (A.4), (A.8) in [1]
        """
        if self.options['asymp'] == AsympOptions.NONE:
            A[0:self.Ns, 0:self.N] = Dk.copy()
        elif self.options['asymp'] == AsympOptions.D:
            A[0:self.Ns, 0:self.N] = Dk.copy()
            A[0:self.Ns, self.N] = self.weight[:, 0].copy()
        else:
            A[0:self.Ns, 0:self.N] = Dk.copy()
            A[0:self.Ns, self.N] = self.weight[:, 0].copy()
            A[0:self.Ns, self.N + 1] = self.weight[:, 0] * self.s
        for m in range(0, self.Nc):
            BB[0:self.Ns, m] = self.weight[:, 0] * self.f[m, :]
        A[self.Ns:2 * self.Ns, :] = A[0:self.Ns, :].imag
        A[0: self.Ns, :] = A[0:self.Ns, :].real
        BB[self.Ns:2 * self.Ns, :] = BB[0:self.Ns, :].imag
        BB[0:self.Ns, :] = BB[0:self.Ns, :].real
        if self.options['asymp'] == AsympOptions.D:
            A[0:self.Ns, self.N] = A[0:self.Ns, self.N].copy()
        elif self.options['asymp'] == AsympOptions.DE:
            A[0:self.Ns, self.N] = A[0:self.Ns, self.N].copy()
            A[self.Ns:2 * self.Ns, self.N + 1] = A[self.Ns:2 * self.Ns, self.N + 1].copy()
        return A, BB

    def plot_magnitude_and_phase(self, fit):
        """
        Plot the magnitude and the phase (if desired) of input f data and new fit data across the input frequencies.
        Repeat for each port.
        :param fit: This is created from the new poles and the calculated residues
        :return:
        """
        freq = self.s / (2 * pi * 1j)
        freq = freq.reshape(1, -1)

        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'

        for plot_num in range(0, self.Nc):
            fig = plt.figure(1, figsize=(8, 7))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(freq[0, :].real, np.abs(self.f[plot_num, :]), color='b', linewidth=1,
                    label='Data {}'.format(plot_num))
            ax.plot(freq[0, :].real, np.abs(fit[:, plot_num]), color='r', linewidth=1, label='FRVF {}'.format(plot_num))
            if self.options['errplot']:
                ax.plot(freq[0, :].real, np.abs(self.f[0] - fit.T[0]), color='g', linewidth=1, label='Deviation')
            plt.xlim(freq[0, 0].real, freq[0, self.Ns - 1].real)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            self.set_scale_legend_options()
            plt.tight_layout()
            plt.savefig('magnitude_plot_{}'.format(plot_num))
            # plt.show()
        if self.options['phaseplot']:
            for plot_num in range(self.Nc):
                fig = plt.figure(1, figsize=(8, 7))
                ax = fig.add_subplot(1, 1, 1)
                plt.xlim(freq[0, 0].real, freq[0, self.Ns - 1].real)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Phase Angle [deg]')
                ax.plot(freq[0, :].real, 180 / pi * np.unwrap(np.angle(self.f[plot_num, :])), color='b', linewidth=1,
                        label='Data {}'.format(plot_num))
                ax.plot(freq[0, :].real, 180 / pi * np.unwrap(np.angle(fit[:, plot_num])), color='r', linewidth=1,
                        label='FRVF {}'.format(plot_num))
                if self.options['errplot']:
                    ax.plot(freq[0, :].real, np.abs((180 / pi * np.unwrap(np.angle(self.f[plot_num, :]))) -
                                                    (180 / pi * np.unwrap(np.angle(fit[:, plot_num])))),
                            color='g', linewidth=1,
                            label='Deviation')
                self.set_scale_legend_options()
                plt.yscale('linear')  # Force linear scale for phase plot
                plt.tight_layout()
                plt.savefig('phase_plot_{}'.format(plot_num))
                # plt.show()

    def plot_sigma(self, LAMBD, C, D):
        """
        This plots the sigma
        :param LAMBD: Diagonal matrix of the initial poles.
        :param C: C matrix
        :param D: D parameter
        :return:
        """
        Dk = np.zeros((self.Ns, self.N), dtype=np.complex)
        for m in range(0, self.N):
            Dk[:, m] = 1 / (self.s - LAMBD[m, m])
        RES3 = D + Dk.dot(C)
        freq = self.s / (2 * pi * 1j)

        freq = freq.reshape(1, -1)

        plt.rcParams['font.size'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.linestyle'] = 'dotted'
        fig = plt.figure(3, figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(freq[0, :], abs(RES3[:, 0].conj().T), color='k', linewidth=1, label='sigma')
        plt.xlim(freq[0, 0], freq[0, self.Ns - 1])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')

        self.set_scale_legend_options()
        plt.tight_layout()
        plt.show()

