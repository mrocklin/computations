from sympy import Symbol, MatrixSymbol, Q, assuming, ZeroMatrix
import numpy as np

from computations.inplace import inplace_compile
from computations.matrices.fortran.core import build
from computations.matrices.blas import GEMM, SYRK
from computations.matrices.lapack import POSV


n = Symbol('n')
m = Symbol('m')
A = MatrixSymbol('A', n, n)
y = MatrixSymbol('y', n, 1)
X = MatrixSymbol('X', n, m)

def test_POSV():
    c = POSV(A, y)
    ic = inplace_compile(c)
    with assuming(Q.real(A), Q.real(y)):
        f = build(ic, [A, y], [A.I*y], modname='posv')

    nA, ny = np.asarray([[2, 1], [1, 2]], dtype='float64').reshape((2, 2)), np.ones(2)
    mA = np.matrix(nA)
    my = np.matrix(ny).T
    expected = np.linalg.solve(mA, my)
    f(nA, ny)
    assert np.allclose(expected, ny)

def test_linear_regression():
    beta = (X.T*X).I * X.T*y

    c = (POSV(X.T*X, X.T*y)
       + SYRK(1.0, X.T, 0.0, ZeroMatrix(m, m))
       + GEMM(1.0, X.T, y, 0.0, ZeroMatrix(n, 1)))

    ic = inplace_compile(c)
    with assuming(Q.real(X), Q.real(y)):
        f = build(ic, [X, y], [beta], modname='linregress')

    nX = np.asarray([[2, 1], [1, 2]], dtype='float64').reshape((2, 2))
    ny = np.ones(2)

    mX = np.matrix(nX)
    my = np.matrix(ny).T
    expected = np.linalg.solve(mX.T*mX, mX.T*my)
    assert np.allclose(expected, f(nX, ny))
