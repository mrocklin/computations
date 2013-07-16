from computations.matrices.shared import detranspose, uplo
from sympy.matrices.expressions import MatrixSymbol

def test_detranspose():
    X = MatrixSymbol('X', 2, 3)
    assert detranspose(X) == X
    assert detranspose(X.T) == X

def test_side():
    try:
        uplo(X, [])
        assert False
    except Exception:
        assert True
