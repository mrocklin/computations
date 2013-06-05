from sympy import MatrixSymbol, Symbol, Q, assuming
from computations.inplace import inplace_compile
from computations.matrices.fortran.mpi import rank_switch

def test_mpi_test_program():
    from computations.matrices.fortran.mpi import mpi_test_program
    s = mpi_test_program('foo')
    assert isinstance(s, str)
    assert 'call foo()' in s

def test_rank_switch():
    d = {0: 'A', 1: 'B', 2: 'C'}
    s = rank_switch(d)
    assert "if (rank .eq. 0)  A()" in s
    assert "if (rank .eq. 1)  B()" in s
    assert "if (rank .eq. 2)  C()" in s
    assert "Need 3 processes" in s
