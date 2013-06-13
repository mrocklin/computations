from computations.matrices.mpi import send, recv, isend, irecv
from computations.matrices.mpi import (Send, Recv, iSend, iRecv, iSendWait,
        iRecvWait)
from computations.matrices.mpi import mpi_key, mpi_tag_key
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

def test_Send():
    s = Send(A, 2)
    assert A in s.inputs
    assert s.dest == 2

def test_Recv():
    r = Recv(A, 1)
    assert A in r.outputs
    assert r.source == 1

def test_iSend():
    s = iSend(A, 2)
    sw = iSendWait(s.request)
    assert A in s.inputs
    assert s.request in s.outputs
    assert s.request in sw.inputs

def test_iRecv():
    r = iRecv(A, 1)
    rw = iRecvWait(A, r.request)
    assert A not in r.outputs
    assert A in rw.outputs
    assert set(rw.inputs).issubset(set(r.outputs))


def test_sendrecv():
    assert A*B+C in s.inputs
    assert r.inputs == ()
    assert A*B+C in r.outputs
    print s.tag, r.tag
    assert s.tag == r.tag

    s2 = send('a', 'c', gemm, axpy)
    assert s.tag != s2.tag

    assert 'mpif' in s.includes
    assert 'mpif' in r.includes
    assert 'mpi' in s.libs
    assert 'mpi' in r.libs

def test_isend():
    s = isend(1, 2, gemm, axpy)
    assert len(s.computations) == 2
    assert A*B+C in s.inputs

def test_irecv():
    s = irecv(1, 2, gemm, axpy)
    assert len(s.computations) == 2
    assert A*B+C in s.outputs


def test_types():
    from computations.matrices.fortran.core import dtype_of
    assert 'int' in dtype_of(s.ierr)
    assert 'int' in dtype_of(r.ierr)
    assert 'int' in dtype_of(r.status)

def streq(a, b):
    canon = lambda s: s.replace(' ', '').lower()
    truth = canon(a) == canon(b)
    if not truth:
        print canon(a)
        print canon(b)
    return truth

def test_send_fortran():
    with assuming(*map(Q.real_elements, (A, B, C))):
        a = s.fortran_call(['A'], ['ierr'])[0]
    b = "call MPI_SEND( A, 15, MPI_DOUBLE_PRECISION, %d, %d, MPI_COMM_WORLD, ierr)"%(s.dest, s.tag)
    assert streq(a, b)

def test_iRecv_fortran():
    r = iRecv(A, 1)
    with assuming(Q.real_elements(A)):
        result = r.fortran_call([], ['A', 'request', 'ierr'])[0]
    expected = 'call MPI_iRECV(A, %d, MPI_DOUBLE_PRECISION, 1, %d, MPI_COMM_WORLD, request, ierr)' % (A.rows*A.cols, r.tag)
    assert streq(result, expected)

def test_iSend_fortran():
    s = iSend(A, 2)
    with assuming(Q.real_elements(A)):
        result = s.fortran_call(['A'], ['request', 'ierr'])[0]
    expected = 'call MPI_iSend(A, %d, MPI_DOUBLE_PRECISION, 2, %d, MPI_COMM_WORLD, request, ierr)' % (A.rows*A.cols, s.tag)
    assert streq(result, expected)

def test_iSendWait_fortran():
    s = iSend(A, 2)
    sw = iSendWait(s.request)
    result = sw.fortran_call(['request'], ['status', 'ierr'])[0]
    expected = 'call MPI_Wait(request, status, ierr)'
    assert streq(result, expected)

def test_iSendWait_fortran():
    r = iRecv(A, 1)
    rw = iRecvWait(A, r.request)
    result = rw.fortran_call(['A', 'request'], ['A', 'status', 'ierr'])[0]
    expected = 'call MPI_Wait(request, status, ierr)'
    assert streq(result, expected)

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

def test_mpi_key():
    from computations.schedule import key_to_cmp
    mpi_cmp = key_to_cmp(mpi_key)
    s = iSend(A, 2)
    sw = iSendWait(s.request)
    assert mpi_cmp(s, gemm)  < 0
    assert mpi_cmp(s, axpy)  < 0
    assert mpi_cmp(gemm, sw) < 0


def test_mpi_tag_key():
    from computations.schedule import key_to_cmp
    mpi_tag_cmp = key_to_cmp(mpi_tag_key)
    s1 = iSend(A, 2, tag=3)
    s2 = iSend(A, 2, tag=4)

    assert mpi_tag_cmp(s1, s2) != 0  # Ties are broken
