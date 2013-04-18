from computations.matrices.fortran.core import *
from sympy import MatrixSymbol, Symbol
from computations.inplace import inplace_compile
from computations.matrices.blas import GEMM

n = Symbol('n')
X = MatrixSymbol('X', n, n)
y = MatrixSymbol('y', n, 1)
inputs = [X, y]
outputs = [X*y]
mathcomp = GEMM(1.0, X, y, 0.0, ZeroMatrix(n, 1))
ic = inplace_compile(mathcomp)
types = {q: 'real*8' for q in [X, y, X*y]}
s = generate(ic, inputs, outputs, types, 'f')

def test_simple():
    assert isinstance(s, str)
    assert "call dgemm('N', 'N', n, 1, n" in s
    assert "1.0" in s
    assert "X, n, y, n,"  in s

def test_dimensions():
    assert set(dimensions(ic)) == set((n, ))
    assert 'integer :: n' in s

def test_dimension_initialization():
    assert dimension_initialization(n, ExprToken(y, 'yvar')) == 'n = size(yvar, 1)'
    assert 'n = size(X, 1)' in s or 'n = size(y, 1)' in s

def test_variable_declaration():
    s = declare_variable_string('a', Symbol('a'), 'integer', True, False)
    assert s == "integer, intent(in) :: a"

    s = declare_variable_string('X', MatrixSymbol('X',n,n), 'real*4', True,True)
    assert s == "real*4, intent(inout) :: X(:,:)"

    s = declare_variable_string('X', MatrixSymbol('X',n,n), 'real*4',
            False,False)
    assert "allocatable" in s

def test_allocate_array():
    from computations.inplace import ExprToken
    v = ExprToken(X, 'Xvar')
    assert allocate_array(v, ['Xvar'], []) == ''
    assert allocate_array(v, [], []) == 'allocate(Xvar(n,n))'
    v = ExprToken(y, 'Yvar')
    assert allocate_array(v, [], []) == 'allocate(Yvar(n))'

def test_allocate_array():
    from computations.inplace import ExprToken
    v = ExprToken(X, 'Xvar')
    assert deallocate_array(v, ['Xvar'], []) == ''
    assert deallocate_array(v, [], []) == 'deallocate(Xvar)'
