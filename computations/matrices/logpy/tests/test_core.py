from computations.matrices.blas import GEMM, SYRK
import computations.matrices.logpy.core

from logpy import run, eq, membero
from logpy.unify import unify, reify
from logpy.variables import variables
from logpy.assoccomm import eq_assoccomm as eqac

from sympy import MatrixSymbol, Symbol

n = Symbol('n')
X = MatrixSymbol('X', n, n)
Y = MatrixSymbol('Y', n, n)
A = MatrixSymbol('_A', n, n)
B = MatrixSymbol('_B', n, n)

gemm = GEMM(1.0, X, Y, 0.0, Y)
syrk = SYRK(1.0, X, 0.0, Y)
vgemm = GEMM(1.0, A, B, 0.0, B)
vsyrk = SYRK(1.0, A, 0.0, B)

def test_unify():
    with variables(A, B):
        s = {A: X, B: Y}
        assert unify(gemm, vgemm, {}) == s
        assert reify(vgemm, s) == gemm
        assert run(1, vgemm, eq(vsyrk, syrk)) == (gemm,)
