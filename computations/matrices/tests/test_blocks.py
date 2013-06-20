from sympy import MatrixSymbol, Symbol, blockcut, BlockMatrix
from computations.matrices.blocks import Slice, Join

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

def test_Join():
    c = Join(B)
    assert c.outputs == (B,)
    assert c.inputs  == (X, Y)

    result = c.fortran_call(["X", "Y"], ["B"])
    print result
    expected = ["B(1:n, 1:m) = X", "B(1:n, m + 1:2*m) = Y"]
    assert set(expected) == set(result)

