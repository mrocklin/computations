from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.matrices.expressions import MatrixSymbol, PermutationMatrix
from sympy.core import Symbol
from sympy import Q

a, b, c, d, x, y, z, n, m, l, k = map(Symbol, 'abcdxyznmlk')

def test_GESV():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, m)
    assert GESV(X, Y).inputs  == (X, Y)
    assert GESV(X, Y).outputs[0].shape == (X.I*Y).shape

def test_POSV():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, m)
    posv = POSV(X, Y)
    assert posv.outputs[0] == X.I*Y
    assert not POSV.valid(posv.inputs, True)
    assert POSV.valid(posv.inputs, Q.symmetric(X) & Q.positive_definite(X))

def test_GESV_codemap():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    codemap = GESV(A, B).codemap('A B IPIV INFO'.split())
    call = GESV.fortran_template % codemap
    assert '(n, m, A, n, IPIV, B, n, INFO)' in call
    assert 'dgesv' in call.lower()

def test_POSV_codemap():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    assumptions = Q.positive_definite(A) & Q.symmetric(A)
    codemap = POSV(A, B).codemap('A B INFO'.split(), assumptions)
    call = POSV.fortran_template % codemap
    assert "('U', n, m, A, n, B, n, INFO)" in call
    assert 'dposv' in call.lower()

def test_LASWP_codemap():
    A = MatrixSymbol('A', n, n)
    ipiv = IPIV(A)
    expr = PermutationMatrix(ipiv)*A
    codemap = LASWP(expr, ipiv).codemap('A IPIV'.split())
    call = LASWP.fortran_template % codemap
    assert "(n, A, n, 1, n, IPIV, 1)" in call
    assert 'dlaswp' in call.lower()
