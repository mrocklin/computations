from computations.matrices.blas import GEMM, SYMM, SYRK, AXPY, COPY
from sympy.matrices.expressions import MatrixSymbol
from sympy.core import Symbol
from sympy import Q, S, ZeroMatrix

a, b, c, d, x, y, z, n, m, l, k = map(Symbol, 'abcdxyznmlk')

def test_GEMM():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    Z = MatrixSymbol('Z', n, n)
    assert GEMM(a, X, Y, b, Z).inputs == (a, X, Y, b, Z)
    assert GEMM(a, X, Y, b, Z).outputs == (a*X*Y+b*Z, )
    assert GEMM(1, X, Y, 0, Y).inputs[0] == 1.0
    assert GEMM(1, X, Y, 0, 0).inputs[4] == ZeroMatrix(X.rows, Y.cols)
    # assert GEMM(1, X, Y, 0, Y).variable_inputs == (X, Y)

def test_GEMM_fortran():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert '_' not in GEMM(1, X, Y, 0, ZeroMatrix(Symbol('_n'),
        Symbol('_m'))).fortran_call([1, 'X', 'Y', 0, 'var'], ['var'])[0]

def test_transpose_GEMM():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    expr = X*Y.T
    c = GEMM(S.One, X, Y.T, S.Zero, Y.T)
    # assert c.variable_inputs == (X, Y, Y.T)
    assert c.inputs == (1, X, Y, 0, Y.T)
    assert c.outputs == (X*Y.T,)

def test_SYRK():
    X = MatrixSymbol('X', n, k)
    Z = MatrixSymbol('Z', n, n)
    Zk = MatrixSymbol('Z', k, k)
    assert SYRK(a, X, b, Z).inputs == (a, X, b, Z)
    assert SYRK(a, X, b, Z).outputs == (a*X*X.T + b*Z, )
    assert SYRK(a, X.T, b, Zk).inputs == (a, X, b, Zk)
    assert SYRK(a, X.T, b, Zk).outputs == (a*X.T*X + b*Zk, )
    # assert SYRK(1, X, 0, X).variable_inputs == (X,)

def test_SYRK_fortran():
    X = MatrixSymbol('X', n, k)
    Z = MatrixSymbol('Z', n, n)
    Zk = MatrixSymbol('Z', k, k)
    assert 'T' not in SYRK(a, X, b, Z).fortran_call(['a', 'X', 'b', 'Z'], ['Z'])[0]
    assert 'T' in SYRK(a, X.T, b, Zk).fortran_call(['a', 'X', 'b', 'Z'], ['Z'])[0]

def test_transpose_SYRK():
    X = MatrixSymbol('X', 3, 3)
    expr = X.T*X
    c = SYRK(S.One, X.T, S.One, X)
    # assert c.variable_inputs == (X,)
    assert c.inputs == (1, X, 1, X)
    assert c.outputs == (X.T*X + X,)

def test_double_transpose_SYRK():
    X = MatrixSymbol('X', 3, 3)
    expr = X.T*X
    c = SYRK(S.One, X.T, S.One, X.T)
    # assert c.variable_inputs == (X.T,)
    assert c.inputs == (1, X.T, 1, X.T)
    assert c.outputs == (X.T*X + X.T,)

def test_valid():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    C = MatrixSymbol('C', n, n)
    assert GEMM.valid((1, A, B, 2, C), True)
    assert not SYMM.valid((1, A, B, 2, C), True)
    assert SYMM.valid((1, A, B, 2, C), Q.symmetric(A))
    assert SYMM.valid((1, A, B, 2, C), Q.symmetric(B))

def test_GEMM_codemap():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, k)
    C = MatrixSymbol('C', n, k)

    codemap = GEMM(a, A, B, c, C).codemap('aABcC')
    call = GEMM.fortran_template % codemap
    assert "('N', 'N', n, k, m, a, A, n, B, m, c, C, n)" in call
    assert 'dgemm' in call.lower()

def test_SYMM_codemap():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', n, m)
    codemap = SYMM(a, A, B, c, C).codemap('aABcC', Q.symmetric(A))
    call = SYMM.fortran_template % codemap
    assert "('L', 'U', n, m, a, A, n, B, n, c, C, n)" in call
    assert 'dsymm' in call.lower()

def test_SYMM_BA():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, n)
    C = MatrixSymbol('C', m, n)
    codemap = SYMM(a, B, A, c, C).codemap('aBAcC', Q.symmetric(A))
    call = SYMM.fortran_template % codemap
    print call
    assert "('R', 'U', m, n, a, A, n, B, m, c, C, m)" in call


def test_SYRK_codemap():
    A = MatrixSymbol('A', n, k)
    C = MatrixSymbol('C', k, k)
    codemap = SYRK(a, A, c, C).codemap('aAcC', )
    call = SYRK.fortran_template % codemap
    assert "('U', 'N', n, k, a, A, n, c, C, k)" in call
    assert 'dsyrk' in call.lower()


def test_AXPY_codemap():
    B = MatrixSymbol('B', n, m)
    C = MatrixSymbol('C', n, m)
    codemap = AXPY(a, B, C).codemap('aBC', True)
    call = AXPY.fortran_template % codemap
    assert "(m*n, a, B, 1, C, 1)" in call
    assert 'daxpy' in call.lower()

def test_COPY_fortran_call():
    X = MatrixSymbol('B', n, m)
    c = COPY(X)
    codemap = c.codemap('XY')
    call = COPY.fortran_template % codemap
    assert "(m*n, X, 1, Y, 1)" in call
    assert 'dcopy' in call.lower()

    s = '\n'.join(c.fortran_call(['X'], ['Y']))
    assert "(m*n, X, 1, Y, 1)" in s
