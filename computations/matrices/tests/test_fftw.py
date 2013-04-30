from sympy import MatrixSymbol, Symbol, Q, assuming
from computations.inplace import inplace_compile
from sympy.matrices.expressions.fourier import DFT
from computations.matrices.fftw import FFTW, Plan

n = Symbol('n')
x = MatrixSymbol('X', n, 1)
c = FFTW(x)

def test_FFTW():
    assert Plan() in c.outputs
    assert DFT(n) * x in c.outputs
    assert x in c.inputs

def test_code_generation():
    from computations.matrices.fortran.core import generate
    ic = inplace_compile(c)
    with assuming(Q.complex_elements(DFT(n)), Q.complex_elements(x)):
        s = generate(ic, [x], [DFT(n)*x])
    with open('tmp.f90','w') as f:
      f.write(s)
    assert 'use, intrinsic :: iso_c_binding' in s
    assert "include 'fftw3.f03'" in s
    assert isinstance(s, str)

