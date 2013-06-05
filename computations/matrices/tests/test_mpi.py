from computations.matrices.mpi import send, recv
from computations.matrices.mpi import Send, Recv
from computations.matrices.blas import GEMM, AXPY
from sympy.matrices.expressions import MatrixSymbol
from sympy import ask, Q, assuming

A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 5)
C = MatrixSymbol('C', 3, 5)
D = MatrixSymbol('D', 3, 5)

gemm = GEMM(1, A, B, 1, C)
axpy = AXPY(2, A*B+C, D)

s = send(1, 2, gemm, axpy)
r = recv(1, 2, gemm, axpy)

def test_sendrecv():
    assert A*B+C in s.inputs
    assert r.inputs == ()
    assert A*B+C in r.outputs
    print s.tag, r.tag
    assert s.tag == r.tag

    s2 = send('a', 'c', gemm, axpy)
    assert s.tag != s2.tag

def test_types():
    from computations.matrices.fortran.core import dtype_of
    assert 'int' in dtype_of(s.ierr)
    assert 'int' in dtype_of(r.ierr)
    assert 'int' in dtype_of(r.status)

def streq(a, b):
    canon = lambda s: s.replace(' ', '')
    return canon(a) == canon(b)

def test_send_fortran():
    with assuming(*map(Q.real_elements, (A, B, C))):
        a = s.fortran_call(['A'], ['ierr'])[0]
        b = "call MPI_SEND( A, 15, MPI_DOUBLE_PRECISION, %d, %d, MPI_COMM_WORLD, ierr)"%(s.dest, s.tag)
        print a
        print b
        assert streq(a, b)

def test_recv_fortran():
    with assuming(*map(Q.real_elements, (A, B, C))):
        a = r.fortran_call([], ['A', 'status', 'ierr'])[0]
        b = "call MPI_RECV( A, 15, MPI_DOUBLE_PRECISION, %d, %d, MPI_COMM_WORLD, status, ierr)"%(r.source, r.tag)
        print a
        print b
        assert streq(a, b)

def test_send_fortran_generate():
    from computations.inplace import inplace_compile
    from computations.matrices.fortran.core import generate
    A = MatrixSymbol('A', 10, 10)
    s = Send(A, 1)
    iss = inplace_compile(s)
    with assuming(Q.real_elements(A)): code = generate(iss, [A], [])
    assert isinstance(code, str)

def test_recv_fortran_generate():
    from computations.inplace import inplace_compile
    from computations.matrices.fortran.core import generate
    A = MatrixSymbol('A', 10, 10)
    r = Recv(A, 2)
    irr = inplace_compile(r)
    with assuming(Q.real_elements(A)): code = generate(irr, [], [A])
    assert isinstance(code, str)
