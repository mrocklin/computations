from computations.matrices.core import MatrixCall
from computations.matrices.shared import (detranspose, trans, LD,
        left_or_right, diag)
from computations.matrices.variables import (alpha, beta, n, m, k, C,
        x, a, b)
from sympy import Q, S, Symbol, Basic
from sympy.matrices.expressions import (MatrixSymbol, MatrixExpr, MatMul,
        Inverse)
from sympy.matrices.expressions.factorizations import UofCholesky
from computations.matrices.permutation import PermutationMatrix
from computations.util import merge
from computations.core import Computation

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
    shape = property(lambda self: (S.One, self.A.shape[0]))

    def fortran_type(self):
        return 'integer'


class LAPACK(MatrixCall):
    """ Linear Algebra PACKage - Dense Matrix computation """
    libs = ["lapack"]

    def typecheck(self):
        return all(map(isinstance, self.args, self.__types__))

class GESV(LAPACK):
    """ General Matrix Vector Solve """
    __types__ = (MatrixExpr, MatrixExpr)
    _inputs   = (A, B)
    _outputs  = (MatMul(PermutationMatrix(IPIV(A.I*B)), A.I, B), IPIV(A.I*B), INFO)
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


    def pseudocode_call(self, inputs_names, outputs_names):
        A, B = inputs_names
        result, ipiv, info = outputs_names
        return ["%(B)s := %(A)s\%(B)s permuted by %(ipiv)s" % locals()]

    @property
    def inputs(self):
        return self.args

    @property
    def outputs(self):
        A, B = self.args
        return (MatMul(PermutationMatrix(IPIV(MatMul(Inverse(A), B))),  Inverse(A), B),
                IPIV(MatMul(Inverse(A), B)),
                INFO)

class GESVLASWP(Computation):
    __types__ = [MatrixExpr, MatrixExpr]
    condition = True
    inplace   = {0: 1}

    def __init__(self, A, B):
        self.inputs = (A, B)
        self.outputs = (MatMul(Inverse(A), B), IPIV(A), INFO)

    def fortran_call(self, input_names, output_names):
        A, B = self.inputs
        PA, IPIV, INFO = self.outputs
        a, b = input_names
        pa, ipiv, info = output_names

        return (GESV(A, B).fortran_call(input_names, [res, ipiv, info]) +
                LASWP(PA, IPIV).fortran_call([pa, ipiv], [pa]))


class LASWP(LAPACK):
    """ Permute rows in a matrix """
    __types__ = (MatMul, PermutationMatrix)
    _inputs   = (MatMul(PermutationMatrix(IPIV(A)), A), IPIV(A))
    _outputs  = (A,)
    inplace   = {0: 0}
    condition = True
    outputs = property(lambda self: (self.inputs[1].A,))

    fortran_template = ("call %(fn)s(%(N)s, %(A)s, %(LDA)s, %(K1)s, %(K2)s, "
                        "%(IPIV)s, %(INCX)s)")

    @property
    def inputs(self):
        return self.args

    @property
    def outputs(self):
        PA, IP = self.args
        A = MatMul(*PA.args[1:])
        return (A,)

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

    def pseudocode_call(self, inputs_names, outputs_names):
        Apermuted, ipiv = inputs_names
        A, = outputs_names
        return ["%(A)s := permute %(A)s by %(ipiv)s" % locals()]

class POSV(LAPACK):
    """ Symmetric Positive Definite Matrix Solve """
    __types__ = (MatrixExpr, MatrixExpr)
    _inputs   = (A, B)
    _outputs  = (A.I*B, UofCholesky(A), INFO)
    inplace   = {0: 1, 1: 0}
    condition = Q.positive_definite(A) & Q.symmetric(A)

    fortran_template = ("call %(fn)s('%(UPLO)s', %(N)s, %(NRHS)s, %(A)s, "
            "%(LDA)s, %(B)s, %(LDB)s, %(INFO)s)\n"
            "  if (%(INFO)s.ne.0) print *, 'POSV failed at ',%(INFO)s\n")

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
        return inputs + (outputs[2],)


    def pseudocode_call(self, inputs_names, outputs_names):
        A, B = inputs_names
        return ["%(B)s := %(A)s\%(B)s" % locals(),
                "%(A)s := Cholesky Decomposition of %(A)s" % locals()]

class POTRS(LAPACK):
    __types__ = (MatrixExpr, MatrixExpr)
    _inputs =  (UofCholesky(A), B)
    _outputs = (A.I*B, INFO)
    inplace = {0: 1}
    condition = True

    fortran_template = ("call %(fn)s('%(UPLO)s', %(N)s, %(NRHS)s, %(A)s, "
            "%(LDA)s, %(B)s, %(LDB)s, %(INFO)s)\n"
            "  if (%(INFO)s.ne.0) print *, 'POTRS failed at ',%(INFO)s\n")

    @property
    def outputs(self):
        from computations.matrices.core import canonicalize
        INFO = self._outputs[-1]
        A, B = self.inputs
        return (canonicalize(A.arg).I*canonicalize(B), INFO)

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

    def pseudocode_call(self, inputs_names, outputs_names):
        A, B = inputs_names
        return ["%(B)s := %(A)s\%(B)s (B is Cholesky factor) " % locals()]
