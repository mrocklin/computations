from computations.matrices.blas import GEMM, SYMM, AXPY, SYRK
from computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from computations.matrices.fftw import FFTW
from computations.matrices.variables import (alpha, beta, n, m, k, A, B, C,
        x, a, b, X, Y, Z)
from sympy import Q, S, ask, Expr, Symbol, Dummy, Integer
from sympy.logic.boolalg import Boolean
from sympy.matrices.expressions import (MatrixExpr, PermutationMatrix,
        MatrixSymbol, ZeroMatrix, MatrixSlice, BlockMatrix)
from sympy.matrices.expressions.fourier import DFT
from computations.core import Identity
from functools import partial

# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (A*A.T, SYRK(1.0, A, 0.0, ZeroMatrix(A.rows, A.rows)), (A,), True),
    (A.T*A, SYRK(1.0, A.T, 0.0, ZeroMatrix(A.cols, A.cols)), (A,), True),
    (alpha*A*B + beta*C, SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B + C, SYMM(alpha, A, B, 1.0, C), (alpha, A, B, C), SYMM.condition),
    (A*B + beta*C, SYMM(1.0, A, B, beta, C), (A, B, beta, C), SYMM.condition),
    (A*B + C, SYMM(1.0, A, B, 1.0, C), (A, B, C), SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), SYMM.condition),
    (A*B, SYMM(1.0, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (A, B), SYMM.condition),

    (alpha*A*B + beta*C, GEMM(*GEMM._inputs), GEMM._inputs, True),
    (alpha*A*B + C, GEMM(alpha, A, B, 1.0, C), (alpha, A, B, C), True),
    (A*B + beta*C, GEMM(1.0, A, B, beta, C), (A, B, beta, C), True),
    (A*B + C, GEMM(1.0, A, B, 1.0, C), (A, B, C), True),
    (alpha*A*B, GEMM(alpha, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), True),
    (A*B, GEMM(1.0, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (A, B), True),

    (alpha*X + Y, AXPY(*AXPY._inputs), AXPY._inputs, AXPY.condition),
    (X + Y, AXPY(1.0, X, Y), (X, Y), True)
]

lapack_patterns = [
    (Z.I*X, POSV(Z, X), (Z, X), Q.symmetric(Z) & Q.positive_definite(Z)),
    (Z.I*X, GESV(Z, X) + LASWP(PermutationMatrix(IPIV(Z.I*X))*Z.I*X, IPIV(Z.I*X)), (Z, X), True),

]

ints = start1, stop1, step1, start2, stop2, step2 = map(Dummy,
        '_start1 _stop1 _step1 _start2 _stop2 _step2'.split())
other_patterns = [
    (DFT(n) * x, FFTW(x), (x, n), True),
]

patterns = lapack_patterns + blas_patterns + other_patterns
