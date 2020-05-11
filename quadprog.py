""" quadprog.py

Author: Jennifer Houle
Date: 4/4/2020

This function quadprog is to replicate the quadprog function in MATLAB. The function used is directly
from [1]. This does not always work to solve the problem. [2] and [3]
were used to replace quadprog, but neither other implementation was able to find solutions on failing data,
though MATLAB did solve the problem.

ML  Py
H == P
f == q
A == G
b == h
meaning that order of args should be the same

[1] divenex, Stack Overflow. Dec. 11, 2019. Accessed on: April 4, 2020.
    [Online]. Available: https://stackoverflow.com/a/59286910.

[2] stephane-caron, "Quadratic Programming in Python". Accessed on: May 3, 2020.
    [Online]. Available: https://scaron.info/blog/quadratic-programming-in-python.html.

[3] nolfwin, GitHub. March 11, 2018. Accessed on: May 3, 2020.
    [Online]. Available: https://github.com/nolfwin/cvxopt_quadprog/blob/master/cvxopt_qp.py.

"""
import cvxopt
import numpy as np
from cvxopt import matrix, solvers


def quadprog(P, q, G=None, h=None, A=None, b=None):
    """
   Quadratic programming problem with both linear equalities and inequalities

       Minimize      0.5 * x @ P @ x + q @ x
       Subject to    G @ x <= h
       and           A @ x = b
    [1]
   """

    P, q = matrix(P), matrix(q)

    if G is not None:
        G, h = matrix(G), matrix(h)

    if A is not None:
        A, b = matrix(A), matrix(b)

    # sol = solvers.qp(P, q, G, h, A, b, solver='mosek')  # requires license
    sol = solvers.qp(P, q, G, h, A, b)

    return np.array(sol['x']).ravel()


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    Quadprog Solver from [2]
    """
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def quadprog2(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Quadprog Solver from [3]
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert (k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert (Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, solver='mosek')

    return np.array(sol['x'])


if __name__ == '__main__':
    from utils import load_file

    matlab = load_file('fourportmay_quadprog',
                       prefix=r"C:\Users\Jenny\Documents\School_Stuff\PhD\ECE504-MC\matrix_fitting_toolbox_copy2")

    H = matlab['H'].astype(np.double)
    ff = matlab['ff'][0].astype(np.double)
    bigB = matlab['bigB'].astype(np.double)
    bigc = matlab['bigc'][0].astype(np.double)
    # dx = quadprog(H, ff, -bigB, bigc.real)
    # dx = quadprog2(H, ff, -bigB, bigc.real)
    dx = cvxopt_solve_qp(H, ff, -bigB, bigc.real)
    print(dx)
