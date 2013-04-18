from sympy import MatrixSymbol, Symbol
from computations.inplace import inplace_compile
from sympy.matrices.expressions.fourier import DFT
from computations.matrices.fftw import FFTW, Plan

n = Symbol('n')
x = MatrixSymbol('X', n, 1)
c = FFTW(x)

types = {q: 'complex(kind=8)' for q in [x, DFT(n), DFT(n)*x]}
types[Plan()] = 'type(C_PTR)'

def test_FFTW():
    assert Plan() in c.outputs
    assert DFT(n) * x in c.outputs
    assert x in c.inputs

def test_code_generation():
    from computations.matrices.fortran.core import generate
    ic = inplace_compile(c)
    s = generate(ic, [x], [DFT(n)*x], types, 'f')

    assert 'use, intrinsic :: iso_c_binding' in s
    assert isinstance(s, str)
