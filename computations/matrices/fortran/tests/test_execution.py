from computations.matrices.fortran.core import build
from computations.matrices.blas import GEMM, SYRK, AXPY, SYMM
from computations.matrices.lapack import POSV
from computations.matrices.elemental import ElemProd
from sympy.matrices.expressions.fourier import DFT
from sympy.matrices.expressions.hadamard import HadamardProduct as HP
from computations.dot import show
from sympy import Symbol, MatrixSymbol, Q, assuming, ZeroMatrix
import numpy as np

n = Symbol('n')
m = Symbol('m')
A = MatrixSymbol('A', n, n)
y = MatrixSymbol('y', n, 1)
X = MatrixSymbol('X', n, m)


def test_SYMM():
    with assuming(Q.real_elements(A), Q.real_elements(X), Q.symmetric(A)):
        f = build(SYMM(1.0, A, X, 0.0, X), [A, X], [A*X], modname='symmtest',
                filename='symmtest.f90')

    nA = np.asarray([[1, 2], [2, 1]], dtype=np.float64, order='F')
    nX = np.asarray([[1], [1]], dtype=np.float64, order='F')
    expected = np.asarray([[3.], [3.]])
    print nX
    f(nA, nX)
    print nX
    print expected
    assert np.allclose(expected, nX)
