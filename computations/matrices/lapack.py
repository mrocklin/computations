from computations.matrices.core import MatrixCall
from computations.matrices.shared import (detranspose, trans, LD,
        left_or_right, diag)
from computations.matrices.variables import (alpha, beta, n, m, k, C,
        x, a, b)
from sympy import Q, S, Symbol, Basic
from sympy.matrices.expressions import (MatrixSymbol, PermutationMatrix,
        MatrixExpr, MatMul)
from computations.util import merge

A = MatrixSymbol('_A', n, n)
B = MatrixSymbol('_B', n, m)

class INFOType(Symbol):
    def fortran_type(self):
        return 'integer'

    name = 'INFO'

INFO = INFOType('INFO')

class IPIV(MatrixExpr):
    def __new__(cls, A):
        return Basic.__new__(cls, A)

    A = property(lambda self: self.args[0])
    shape = property(lambda self: (1, self.A.shape[0]))

class LAPACK(MatrixCall):
    """ Linear Algebra PACKage - Dense Matrix computation """
    flags = ["-llapack"]

class GESV(LAPACK):
    """ General Matrix Vector Solve """
    _inputs   = (A, B)
    _outputs  = (PermutationMatrix(IPIV(A.I*B))*A.I*B, IPIV(A.I*B), INFO)
    inplace   = {0: 1}
    condition = True  # TODO: maybe require S to be invertible?

    fortran_template = ("call %(fn)s(%(N)s, %(NRHS)s, %(A)s, "
                        "%(LDA)s, %(IPIV)s, %(B)s, %(LDB)s, %(INFO)s)")

    @staticmethod
    def arguments(inputs, outputs):
        return inputs + outputs[1:]

    def codemap(self, names, assumptions=()):
        varnames = 'A B IPIV INFO'.split()
        A, B = self.args
        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'LDB': LD(B),
                 'N': str(A.shape[0]),
                 'NRHS': str(B.shape[1]),
                 'fn': self.fnname(self.typecode)}
        return merge(namemap, other)


class LASWP(LAPACK):
    """ Permute rows in a matrix """
    _inputs   = (PermutationMatrix(IPIV(A))*A, IPIV(A))
    _outputs  = (A,)
    inplace   = {0: 0}
    condition = True
    outputs = property(lambda self: (self.inputs[1].A,))

    fortran_template = ("call %(fn)s(%(N)s, %(A)s, %(LDA)s, %(K1)s, %(K2)s, "
                        "%(IPIV)s, %(INCX)s)")

    def codemap(self, names, assumptions=()):
        varnames = 'A IPIV'.split()

        Q, IPIV = self.args
        assert (isinstance(Q, MatMul) and
                isinstance(Q.args[0], PermutationMatrix))
        A = MatMul(*Q.args[1:])  # A is everything other than the permutation

        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'K1': str(1),
                 'K2': str(A.rows),
                 'N': str(A.cols),
                 'INCX': str(1),
                 'fn': self.fnname(self.typecode)}
        return merge(namemap, other)


class POSV(LAPACK):
    """ Symmetric Positive Definite Matrix Solve """
    _inputs   = (A, B)
    _outputs  = (A.I*B, INFO)
    inplace   = {0: 1}
    condition = Q.positive_definite(A) & Q.symmetric(A)

    fortran_template = ("call %(fn)s('%(UPLO)s', %(N)s, %(NRHS)s, %(A)s, "
                        "%(LDA)s, %(B)s, %(LDB)s, %(INFO)s)")

    def codemap(self, names, assumptions=()):
        varnames = 'A B INFO'.split()
        A, B = self.args
        namemap  = dict(zip(varnames, names))
        other = {'LDA': LD(A),
                 'LDB': LD(B),
                 'N': str(A.shape[0]),
                 'NRHS': str(B.shape[1]),
                 'UPLO': 'U',
                 'fn': self.fnname(self.typecode)}
        return merge(namemap, other)

    @staticmethod
    def arguments(inputs, outputs):
        return inputs + (outputs[1],)
