from computations.matrices.fortran.core import *
from sympy import MatrixSymbol, Symbol, Q
from computations.inplace import inplace_compile
from computations.matrices.blas import GEMM, AXPY

n, m, k, l = map(Symbol, 'nmkl')
X = MatrixSymbol('X', n, n)
Y = MatrixSymbol('Y', n, n)
y = MatrixSymbol('y', n, 1)
inputs = [X, y]
outputs = [X*y]
mathcomp = GEMM(1.0, X, y, 0.0, ZeroMatrix(n, 1))
ic = inplace_compile(mathcomp)
with assuming(Q.real_elements(X), Q.real_elements(y)):
    s = generate(ic, inputs, outputs, name='f')
    f2py = generate_f2py_header(ic, inputs, outputs, name='f')
    mod = generate_module(ic, inputs, outputs, name='f', modname='mod')

def test_simple():
    assert isinstance(s, str)
    assert "call dgemm('n', 'n', n, 1, n" in s.lower()
    assert "1.0" in s
    assert "x, n, y, n,"  in s.lower()

def test_dimensions():
    assert set(dimensions(ic)) == set((n, ))
    assert 'integer :: n' in s

    c = inplace_compile(AXPY(1.0, MatrixSymbol('X', n, 2*m), MatrixSymbol('Y',
        n, 2*m)))
    assert set(dimensions(c)) == set((n, m))


def test_dimension_initialization():
    assert dimension_initialization(n, ExprToken(y, 'yvar')) == 'n = size(yvar, 1)'
    assert 'n = size(X, 1)' in s or 'n = size(y, 1)' in s

    Z = MatrixSymbol('Z', 2*k, 2*l)
    assert dimension_initialization(l, ExprToken(Z, 'Z')).replace(' ', '') == \
            'l=size(Z,2)/2'

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
    assert 'integer' in dtype_of(X, Q.integer_elements(X))
    assert 'real' in dtype_of(X, Q.real_elements(X))
    assert 'complex' in dtype_of(X, Q.complex_elements(X))

    alpha = Symbol('alpha', integer=True)
    assert 'integer' in dtype_of(alpha)

def test_f2py():
    assert "X(n,n)" in f2py
    assert 'integer, intent(in) :: n' in f2py
    assert 'n' in f2py.strip().split('\n')[0]  # n in first line
    assert not any('call' in line and 'n' in line for line in f2py.split('\n'))

def test_f2py_compile():
    with open('tmp/tmp.f90', 'w') as f:
        f.write(mod)
    import os
    pipe = os.popen('f2py -c tmp/tmp.f90 -m mod -lblas -llapack')
    text = pipe.read()
    if "Error" in text:
        print text
        assert False

def test_build():
    with assuming(Q.real_elements(X), Q.real_elements(y)):
        f = build(ic, inputs, outputs, modname='build')

def test_numerics():
    import numpy as np
    with assuming(Q.real_elements(X), Q.real_elements(y)):
        f = build(ic, inputs, outputs, filename='numerics.f90',
                modname='numerics')
    nX, ny = np.ones((5, 5)), np.ones(5)
    result = np.matrix(nX) * np.matrix(ny).T
    assert np.allclose(f(nX, ny), result)

def test_tokens_of():
    gemm = GEMM(1, X, Y, 0, Y)
    igemm = inplace_compile(gemm)
    (computations, vars, input_tokens, input_vars, output_tokens, tokens,
                        dimens) = tokens_of(igemm, [X, Y], [X*Y])

    assert list(input_tokens) == ['X', 'Y']

def test_numel():
    A = MatrixSymbol('A', 2, 3)
    x = Symbol('x')
    assert numel(A) == 2*3
    assert numel(x) == 1

def test_nbytes():
    A = MatrixSymbol('A', 2, 3)
    x = Symbol('x')
    with assuming(Q.real_elements(A), Q.real(x)):
        assert nbytes(A) == 2*3*8
        assert nbytes(x) == 8

def test_front_flag():
    assert all(front_flag(s) for s in
            ['-lblas', '-L/usr/lib', '-I/usr/include'])

def test_var_that_uses_dimension():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('X', m, k)
    Z = MatrixSymbol('Z', 2*k, 2*l)
    ets = ExprToken(X, 'X'), ExprToken(Y, 'Y'), ExprToken(Z, 'Z')
    assert var_that_uses_dimension(n, ets) == ExprToken(X, 'X')
    assert var_that_uses_dimension(k, ets) == ExprToken(Y, 'Y')
    assert var_that_uses_dimension(l, ets) == ExprToken(Z, 'Z')
