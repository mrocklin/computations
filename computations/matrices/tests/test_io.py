from computations.matrices.io import ReadFromFile, WriteToFile
from sympy.matrices.expressions import MatrixSymbol

A = MatrixSymbol('A', 3, 4)

def test_read():
    r = ReadFromFile('filename.txt', A)
    assert not r.inputs
    assert A in r.outputs

def test_write():
    w = WriteToFile('filename.txt', A)
    assert A in w.inputs
    assert A not in w.outputs
