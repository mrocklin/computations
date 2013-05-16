from sympy import Symbol, MatrixSymbol
from computations.matrices.blas import GEMM
from computations.profile import Profile
n = Symbol('n')
X = MatrixSymbol('X', n, n)
Y = MatrixSymbol('Y', n, n)
gemm = GEMM(1, X, Y, 0, Y)
pgemm = Profile(gemm)

def test_Profile():
    assert pgemm.inputs == gemm.inputs
    assert len(pgemm.outputs) > len(gemm.outputs)
    assert set(pgemm.time_vars).issubset(pgemm.outputs)
    assert set(gemm.outputs).issubset(pgemm.outputs)

    s = pgemm.fortran_call("1 X Y 0 Y".split(), "duration rate max start end Y".split())

    assert isinstance(s, str)

