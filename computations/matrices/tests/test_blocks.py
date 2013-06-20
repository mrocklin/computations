from sympy import MatrixSymbol, Symbol, blockcut, BlockMatrix, assuming, Q
from computations.matrices.blocks import Slice, Join
from computations.matrices.fortran.core import build
import numpy as np

n = Symbol('n')
m = Symbol('m')
X = MatrixSymbol('X', n, m)
Y = MatrixSymbol('Y', n, m)

B = BlockMatrix([[X, Y]])

def test_Slice():
    Xs = X[10:20, 40:50]
    c = Slice(Xs)
    assert c.outputs == (Xs,)
    assert c.inputs  == (X,)

    result = c.fortran_call(["X"], ["Y"])
    assert result == ["Y = X(11:20, 41:50)"]

def test_Slice_build():
    c = Slice(X[:2, :2])
    with assuming(Q.real_elements(X)):
        f = build(c, [X], [X[:2, :2]], filename='tmp/slice.f90', modname='slice')
    nX = np.eye(3, dtype=np.float64)
    result   = f(nX)
    expected = np.eye(2, dtype=np.float64)

    assert np.allclose(result, expected)

def test_Join():
    c = Join(B)
    assert c.outputs == (B,)
    assert c.inputs  == (X, Y)

    result = c.fortran_call(["X", "Y"], ["B"])
    print result
    expected = ["B(1:n, 1:m) = X", "B(1:n, m + 1:2*m) = Y"]
    assert set(expected) == set(result)

def test_Join_build():
    c = Join(B)
    with assuming(Q.real_elements(X), Q.real_elements(Y)):
        f = build(c, [X, Y], [B], filename='tmp/join.f90', modname='join')
    nX = np.ones((2, 3), dtype=np.float64)
    nY = np.ones((2, 3), dtype=np.float64)
    result   = f(nX, nY)
    expected = np.ones((2, 6), dtype=np.float64)

    assert np.allclose(result, expected)
