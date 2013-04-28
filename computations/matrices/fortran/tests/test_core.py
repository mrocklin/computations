from computations.matrices.fortran.core import *
from sympy import MatrixSymbol, Symbol, Q
from computations.inplace import inplace_compile
from computations.matrices.blas import GEMM

n = Symbol('n')
X = MatrixSymbol('X', n, n)
y = MatrixSymbol('y', n, 1)
inputs = [X, y]
outputs = [X*y]
mathcomp = GEMM(1.0, X, y, 0.0, ZeroMatrix(n, 1))
ic = inplace_compile(mathcomp)
with assuming(Q.real(X), Q.real(y)):
    s = generate(ic, inputs, outputs, name='f')
    f2py = generate_f2py_header(ic, inputs, outputs, name='f')
    mod = generate_module(ic, inputs, outputs, name='f', modname='mod')

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
    s = declare_variable_string('a', Symbol('a'), 'integer', True, False, False)
    assert s == "integer, intent(in) :: a"

    s = declare_variable_string('X', MatrixSymbol('X',n,n), 'real*4', True,
            True, True)
    assert s == "real*4, intent(inout) :: X(:,:)"

    s = declare_variable_string('X', MatrixSymbol('X',n,n), 'real*4',
            False, True, False)
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

def test_dtype_of():
    X = MatrixSymbol('X', n, n)
    assert 'integer' in dtype_of(X, Q.integer(X))
    assert 'real' in dtype_of(X, Q.real(X))
    assert 'complex' in dtype_of(X, Q.complex(X))

def test_f2py():
    assert "X(n,n)" in f2py
    assert 'integer, intent(in) :: n' in f2py
    assert 'n' in f2py.strip().split('\n')[0]  # n in first line
    assert not any('call' in line and 'n' in line for line in f2py.split('\n'))

def test_f2py_compile():
    with open('tmp.f90', 'w') as f:
        f.write(mod)
    import os
    pipe = os.popen('f2py -c tmp.f90 -m mod -lblas -llapack')
    text = pipe.read()
    if "Error" in text:
        print text
        assert False

def test_build():
    with assuming(Q.real(X), Q.real(y)):
        f = build(ic, inputs, outputs)

def test_numerics():
    import numpy as np
    with assuming(Q.real(X), Q.real(y)):
        f = build(ic, inputs, outputs)
    nX, ny = np.ones((5, 5)), np.ones(5)
    result = np.matrix(nX) * np.matrix(ny).T
    assert np.allclose(f(nX, ny), result)
