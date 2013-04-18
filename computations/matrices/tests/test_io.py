from computations.matrices.io import send, recv
from computations.matrices.blas import GEMM, AXPY
from sympy.matrices.expressions import MatrixSymbol
from sympy import S

A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 5)
C = MatrixSymbol('C', 3, 5)
D = MatrixSymbol('D', 3, 5)

gemm = GEMM(S(1), A, B, S(1), C)
axpy = AXPY(S(2), A*B+C, D)

def test_sendrecv():
    s = send('a', 'b', gemm, axpy)
    r = recv('a', 'b', gemm, axpy)
    assert s.inputs == (A*B+C,)
    assert s.outputs == ()
    assert r.inputs == ()
    assert r.outputs == (A*B+C,)
    assert s.tag == r.tag

    s2 = send('a', 'c', gemm, axpy)
    assert s.tag != s2.tag

