from sympy import MatrixSymbol, Symbol, Q, assuming
from computations.inplace import inplace_compile
from computations.matrices.fortran.mpi import rank_switch

def test_generate_mpi_program():
    from computations.matrices.io import ReadFromFile, WriteToFile
    from computations.matrices.fortran.mpi import generate_mpi_tester
    A = MatrixSymbol('A', 10, 10)
    comp = ReadFromFile('tmp/input.txt', A) + WriteToFile('tmp/output.txt', A)
    icomp = inplace_compile(comp)
    with assuming(Q.real_elements(A)):
        s = generate_mpi_tester(icomp, [], [])
    assert isinstance(s, str)

def test_rank_switch():
    d = {0: 'A', 1: 'B', 2: 'C'}
    s = rank_switch(d)
    assert "if (rank .eq. 0)  A()" in s
    assert "if (rank .eq. 1)  B()" in s
    assert "if (rank .eq. 2)  C()" in s
    assert "Need 3 processes" in s
