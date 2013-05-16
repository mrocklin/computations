from sympy import Symbol, MatrixSymbol, Q, assuming
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
    assert pgemm.inplace == {5: 4}
    assert pgemm.libs == gemm.libs
    assert pgemm.includes == gemm.includes

def test_execution():
    from computations.matrices.fortran.core import build
    with assuming(Q.real_elements(X), Q.real_elements(Y)):
        f = build(pgemm, [X, Y], [pgemm.duration])
    assert callable(f)
    import numpy as np
    nX, nY = np.random.rand(500, 500), np.random.rand(500, 500)
    result = f(nX, nY)
    assert isinstance(result, float)
    assert result > 0
