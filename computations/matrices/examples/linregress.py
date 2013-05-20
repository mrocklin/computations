""" Linear Regression """

from sympy import MatrixSymbol, Q, ZeroMatrix, assuming
from sympy import Symbol
n = Symbol('n')
m = Symbol('m')
X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)
beta = (X.T*X).I * X.T*y

from computations.matrices.blas import SYRK, GEMM
from computations.matrices.lapack import POSV
c = (POSV(X.T*X, X.T*y)
   + SYRK(1.0, X.T, 0.0, ZeroMatrix(m, m))
   + GEMM(1.0, X.T, y, 0.0, ZeroMatrix(n, 1)))

assumptions = Q.fullrank(X), Q.real_elements(X), Q.real_elements(y)
