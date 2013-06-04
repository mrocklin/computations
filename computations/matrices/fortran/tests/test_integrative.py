from sympy import Symbol, MatrixSymbol, Q, assuming, ZeroMatrix
import numpy as np

from computations.matrices.fortran.core import build
from computations.matrices.blas import GEMM, SYRK, AXPY, SYMM
from computations.matrices.lapack import POSV
from computations.matrices.fftw import FFTW, IFFTW
from computations.matrices.elemental import ElemProd
from sympy.matrices.expressions.fourier import DFT
from sympy.matrices.expressions.hadamard import HadamardProduct as HP
from computations.dot import show

n = Symbol('n')
m = Symbol('m')
A = MatrixSymbol('A', n, n)
y = MatrixSymbol('y', n, 1)
X = MatrixSymbol('X', n, m)

def test_es_toy():
    K = MatrixSymbol('K',n,1)
    phi = MatrixSymbol('phi', n, 1)
    V = MatrixSymbol('V',n,1)

    c = AXPY(1.0, HP(K,phi), DFT(n).T*HP(V,DFT(n)*phi)) + ElemProd(K,phi) + IFFTW(HP(V, DFT(n) * phi)) + FFTW(phi) + ElemProd(V, DFT(n) * phi)
#    show(c)
    with assuming(Q.complex_elements(phi), Q.real_elements(K), Q.real_elements(V)):
        f = build(c, [K,V,phi], [HP(K,phi) + DFT(n).T * HP(V,DFT(n) * phi)], modname='es', filename='es.f90')

def test_POSV():
    c = POSV(A, y)
    with assuming(Q.real_elements(A), Q.real_elements(y)):
        f = build(c, [A, y], [A.I*y], modname='posv', filename='posv.f90')

    nA, ny = np.asarray([[2, 1], [1, 2]], dtype='float64').reshape((2, 2)), np.ones(2)
    mA = np.matrix(nA)
    my = np.matrix(ny).T
    expected = np.linalg.solve(mA, my)
    f(nA, ny)
    assert np.allclose(expected, ny)

def test_linear_regression():
    beta = (X.T*X).I * X.T*y

    c = (POSV(X.T*X, X.T*y)
       + SYRK(1.0, X.T, 0.0, ZeroMatrix(m, m))
       + GEMM(1.0, X.T, y, 0.0, ZeroMatrix(n, 1)))

    with assuming(Q.real_elements(X), Q.real_elements(y)):
        f = build(c, [X, y], [beta],
                    modname='linregress', filename='linregress.f90')

    nX = np.asarray([[1, 2], [3, 4], [5, 7]], dtype='float64', order='F')
    ny = np.asarray([[1], [1], [1]], dtype='float64', order='F')

    mX = np.matrix(nX)
    my = np.matrix(ny)
    expected = np.asarray(np.linalg.solve(mX.T*mX, mX.T*my)).squeeze()
    result = f(nX, ny)
    assert np.allclose(expected, result)

def test_fftw():
    c = FFTW(y)
    with assuming(Q.complex_elements(y)):
        f = build(c, [y], [DFT(y)], modname='fftw', filename='fftw.f90')

    x = np.zeros(8, dtype='complex')
    expected = np.fft.fft(x)
    f(x)
    assert np.allclose(expected, x)

def test_fftw_inverse():
    c = FFTW(y)
    with assuming(Q.complex(y)):
        f = build(c, [y], [DFT(y)], modname='fftw2', filename='fftw2.f90')

    c = IFFTW(y)
    with assuming(Q.complex(y)):
        fi = build(c, [y], [DFT(y).T], modname='ifftw', filename='ifftw.f90')

    x = np.random.random_sample((8,)) + 1j * np.random.random_sample((8,))
    expected = x
    f(x)
    fi(x)
    assert np.allclose(expected, x)

def test_SYMM():
    with assuming(Q.real_elements(A), Q.real_elements(X), Q.symmetric(A)):
        f = build(SYMM(1.0, A, X, 0.0, ZeroMatrix(A.rows, X.cols)),
                [A, X], [A*X], modname='symmtest', filename='symmtest.f90')

    nA = np.asarray([[1, 2], [2, 1]], dtype=np.float64, order='F')
    nX = np.asarray([[1], [1]], dtype=np.float64, order='F')
    expected = np.asarray([[3.], [3.]])
    result = f(nA, nX)
    assert np.allclose(expected, result)

def test_GEMM():
    with assuming(Q.real_elements(A), Q.real_elements(X)):
        f = build(GEMM(1.0, A, X, 0.0, ZeroMatrix(A.rows, X.cols)),
                [A, X], [A*X], modname='gemmtest', filename='gemmtest.f90')

    nA = np.asarray([[1, 2], [3, 4]], dtype=np.float64, order='F')
    nX = np.asarray([[1, 1], [1, 1]], dtype=np.float64, order='F')
    expected = np.asarray([[3., 3.], [7., 7.]])
    result = f(nA, nX)
    assert np.allclose(expected, result)

def test_WriteToFile():
    from computations.matrices.io import WriteToFile
    filename = 'test_write.dat'
    with assuming(Q.real_elements(X)):
        f = build(WriteToFile(filename, X), [X], [],
                modname='writetest', filename='writetest.f90')

    data = np.asarray([[1., 2., 3.], [4., 5., 6.]])
    result = f(data)
    with open(filename) as f:
        assert "1.0" in f.read()

def test_ReadFromFile():
    from computations.matrices.io import ReadFromFile
    X = MatrixSymbol('X', 2, 3)
    with assuming(Q.real_elements(X)):
        f = build(ReadFromFile('test_read.dat', X), [], [X],
                modname='readtest', filename='readtest.f90')

    print "hello World!"
    result = f()
    print result
    assert result[0, 0] == 1.0

def test_ReadWrite():
    from computations.matrices.io import ReadFromFile, WriteToFile
    X = MatrixSymbol('X', 2, 3)
    c = ReadFromFile('test_read.dat', X) + WriteToFile('test_write2.dat', X)
    with assuming(Q.real_elements(X)):
        f = build(c, [], [], modname='readwritetest',
                             filename='readwritetest.f90')
    f()
    with open('test_read.dat') as f:
        with open('test_write2.dat') as g:
            assert f.read() == g.read()
