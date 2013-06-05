from sympy import MatrixSymbol, Symbol, Q, assuming
from computations.inplace import inplace_compile

def test_generate_mpi_program():
    from computations.matrices.io import ReadFromFile, WriteToFile
    from computations.matrices.fortran.mpi import generate_mpi_tester
    A = MatrixSymbol('A', 10, 10)
    comp = ReadFromFile('tmp/input.txt', A) + WriteToFile('tmp/output.txt', A)
    icomp = inplace_compile(comp)
    with assuming(Q.real_elements(A)):
        s = generate_mpi_tester(icomp, [], [])
    assert isinstance(s, str)
