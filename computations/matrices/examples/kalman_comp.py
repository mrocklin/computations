from computations.matrices.examples.kalman import (newmu, newSigma,
        assumptions, mu, Sigma, R, H, data, n, k)

from computations.matrices.blas import GEMM, SYMM
from computations.matrices.lapack import POSV, POTRS

from sympy.matrices.expressions import ZeroMatrix, Transpose
from sympy.matrices.expressions.factorizations import UofCholesky
from sympy import assuming, Q

def make_kalman_comp(mu, Sigma, H, R, data, assumptions=assumptions):
    Z = H*Transpose(H*Sigma) + R
    A = Z.I * (-1.0*data + H*mu)
    B = Z.I * H

    with assuming(*assumptions):
        c = (
             SYMM(1.0, H, Sigma, 0.0, ZeroMatrix(*H.shape)) +
             GEMM(1.0, H, mu, -1.0, data) +
             GEMM(1.0, H, Transpose(H*Sigma), 1.0, R) +
             POSV(Z, H) +
             POTRS(UofCholesky(Z), -1.0 * data + H*mu) +

             GEMM(1.0, H.T, A, 0.0, ZeroMatrix(*(H.T*A).shape)) +
             SYMM(1.0, Sigma, H.T*A, 1.0, mu) +

             GEMM(1.0, H.T, B, 0.0, ZeroMatrix(*(H.T*B).shape)) +
             SYMM(1.0, H.T*B, Sigma, 0.0, ZeroMatrix(*Sigma.shape)) +
             SYMM(-1.0, Sigma, H.T*B*Sigma, 1.0, Sigma))  # careful, overwrite
    return c

c = make_kalman_comp(mu, Sigma, H, R, data, assumptions)
