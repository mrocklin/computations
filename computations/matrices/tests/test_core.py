from computations.matrices.core import remove_numbers, canonicalize
from computations.matrices.blas import GEMM
from sympy.matrices.expressions import MatrixSymbol, MatAdd
from sympy.core import Symbol, S, Float

def test_remove_numbers():
    X = MatrixSymbol('X', 1, 3)
    x = Symbol('x')
    assert remove_numbers([x, X, 1, 1.0, S.One]) == [x, X]

def test_inplace():
    a = Symbol('a')
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    g = GEMM(a, X, Y, S.Zero, Y)
    assert g.inplace == {0: 4}

def test_canonicalize():
    X = MatrixSymbol('X', 3, 3)
    assert canonicalize(MatAdd(X, X)) == 2*X

def test_sympify_floats():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    g = GEMM(2.0, X, Y, 0.0, Y)
    assert isinstance(g.args[0], Float)

def test_typecheck():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    assert GEMM(2.0, X, Y, 0.0, Y).typecheck()
    assert not GEMM(2.0, 1, Y, 0.0, Y).typecheck()
    assert not GEMM(2.0, X, Y, X, Y).typecheck()
