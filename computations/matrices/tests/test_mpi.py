from computations.matrices.mpi import send, recv
from computations.matrices.blas import GEMM, AXPY
from sympy.matrices.expressions import MatrixSymbol

A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 5)
C = MatrixSymbol('C', 3, 5)
D = MatrixSymbol('D', 3, 5)

gemm = GEMM(1, A, B, 1, C)
axpy = AXPY(2, A*B+C, D)

def test_sendrecv():
    s = send('a', 'b', gemm, axpy)
    r = recv('a', 'b', gemm, axpy)
    assert A*B+C in s.inputs
    assert r.inputs == ()
    assert A*B+C in r.outputs
    print s.tag, r.tag
    assert s.tag == r.tag

    s2 = send('a', 'c', gemm, axpy)
    assert s.tag != s2.tag
