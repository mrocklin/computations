from computations.matrices.io import ReadFromFile, WriteToFile, disk_io
from sympy.matrices.expressions import MatrixSymbol

A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 5)

def test_read():
    r = ReadFromFile('filename.txt', A)
    assert not r.inputs
    assert A in r.outputs

def test_write():
    w = WriteToFile('filename.txt', A)
    assert A in w.inputs
    assert A not in w.outputs

def test_disk_io():
    from computations.matrices.blas import GEMM
    from sympy import ZeroMatrix
    gemm = GEMM(1, A, B, 0, ZeroMatrix(3, 5))
    filenames = {A: 'A.dat', B: 'B.dat', A*B: 'AB.dat'}
    io_gemm = disk_io(gemm, filenames) + gemm
    print io_gemm
    assert not {A, B, A*B}.intersection(io_gemm.inputs)
    assert not {A, B, A*B}.intersection(io_gemm.outputs)
