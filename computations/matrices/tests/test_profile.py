from sympy import Symbol, MatrixSymbol, Q, assuming
from computations.matrices.blas import GEMM
from computations.profile import Profile, ProfileMPI
from computations.core import CompositeComputation
from computations.matrices.fortran.core import build
import numpy as np
n = Symbol('n')
X = MatrixSymbol('X', n, n)
Y = MatrixSymbol('Y', n, n)
gemm = GEMM(1, X, Y, 0, Y)
pgemm = Profile(gemm)
mpipgemm = ProfileMPI(gemm)

def test_Profile():
    for p in pgemm, mpipgemm:
        assert p.inputs == gemm.inputs
        assert len(p.outputs) > len(gemm.outputs)
        assert set(p.time_vars).issubset(p.outputs)
        assert set(gemm.outputs).issubset(p.outputs)
        assert set(p.libs).issuperset(gemm.libs)
        assert set(p.includes).issuperset(gemm.includes)

def test_just_Profile():
    s = pgemm.fortran_call("1 X Y 0 Y".split(), "duration rate max start end Y".split())
    assert isinstance(s, str)
    assert pgemm.inplace == {5: 4}

def test_just_ProfileMPI():
    s = mpipgemm.fortran_call("1 X Y 0 Y".split(), "duration start end Y".split())
    assert isinstance(s, str)
    assert mpipgemm.inplace == {3: 4}
    assert 'mpi' in mpipgemm.libs
    assert 'mpif' in mpipgemm.includes

def test_execution():
    with assuming(Q.real_elements(X), Q.real_elements(Y)):
        f = build(pgemm, [X, Y], [pgemm.duration], filename='profile.f90',
                modname='profile')
    assert callable(f)
    nX, nY = np.random.rand(500, 500), np.random.rand(500, 500)
    result = f(nX, nY)
    assert isinstance(result, float)
    assert result > 0

def test_linregress():
    from computations.matrices.examples.linregress import c, assumptions, X, y
    cc = CompositeComputation(*map(Profile, c.toposort()))
    with assuming(*assumptions):
        f = build(cc, [X, y], [comp.duration for comp in cc.toposort()],
                filename='profile_linregress.f90', modname='profile_linregress')
    nX, ny = np.random.rand(500, 500), np.random.rand(500, 1)
    t1, t2, t3 = f(nX, ny)
    assert all(isinstance(t, float) for t in (t1, t2, t3))
