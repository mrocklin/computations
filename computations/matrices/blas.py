from computations.matrices.core import (MatrixCall, remove_numbers,
        is_number, defloat)
from computations.util import unique
from computations.inplace import Copy
from computations.matrices.shared import (detranspose, trans, LD,
        left_or_right, diag)
from computations.matrices.variables import (alpha, beta, n, m, k, A, B, C, D,
        x, a, b, X, Y)
from sympy import Q, S
from sympy.utilities.iterables import dict_merge as merge
from sympy.matrices.expressions import ZeroMatrix, Transpose

class BLAS(MatrixCall):
    """ Basic Linear Algebra Subroutine - Dense Matrix computation """
    libs = ["blas"]

class MM(BLAS):
    """ Matrix Multiply """
    def __init__(self, alpha, A, B, beta, C, typecode='D'):
        if isinstance(C, ZeroMatrix):
            C = ZeroMatrix(A.rows, B.cols)
        if isinstance(alpha, int):    alpha = float(alpha)
        if isinstance(beta, int):     beta = float(beta)

        super(MM, self).__init__(alpha, A, B, beta, C, typecode)

    _inputs   = (alpha, A, B, beta, C)
    _outputs  = (alpha*A*B + beta*C,)
    inplace   = {0: 4}
    condition = True

    @property
    def inputs(self):
        alpha, A, B, beta, C = self.args
        if isinstance(C, ZeroMatrix):    # special case this
            C = ZeroMatrix(A.rows, B.cols)
        # Sometimes we use C only as an output. It should be detransposed
        A = detranspose(A)
        B = detranspose(B)
        return alpha, A, B, beta, C

    @property
    def outputs(self):
        alpha, A, B, beta, C = self.args
        if isinstance(C, ZeroMatrix):    # special case this
            C = ZeroMatrix(A.rows, B.cols)
        return (alpha*A*B + beta*C,)

    def codemap(self, names, assumptions=True):
        varnames = 'alpha A B beta C'.split()
        alpha, A, B, beta, C = self.args
        if is_number(names[0]):     names[0] = float(names[0])
        if is_number(names[3]):     names[3] = float(names[3])

        namemap  = dict(zip(varnames, names))
        other = {'TRANSA': trans(A), 'TRANSB': trans(B),
                 'LDA': LD(A), 'LDB': LD(B), 'LDC': LD(C),
                 'M':str(A.shape[0]), 'K':str(B.shape[0]), 'N':str(B.shape[1]),
                 'fn': self.fnname(self.typecode),
                 'SIDE': left_or_right(A, B, Q.symmetric, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': 'U'} # TODO: symmetric matrices might be stored low
        return merge(namemap, other)


class GEMM(MM):
    """ General Matrix Multiply """
    fortran_template = ("call %(fn)s('%(TRANSA)s', '%(TRANSB)s', "
                        "%(M)s, %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s, %(beta)s, %(C)s, %(LDC)s)")

class SYMM(MM):
    """ Symmetric Matrix Multiply """
    condition = Q.symmetric(A) | Q.symmetric(B)
    fortran_template = ("call %(fn)s('%(SIDE)s', '%(UPLO)s', %(M)s, %(N)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, %(B)s, %(LDB)s, "
                        "%(beta)s, %(C)s, %(LDC)s)")

class AXPY(BLAS):
    """ Matrix Matrix Addition `alpha X + Y` """
    _inputs   = (alpha, X, Y)
    _outputs  = (alpha*X + Y,)
    inplace   = {0: 2}
    condition = True

    fortran_template = ("call %(fn)s(%(N)s, %(alpha)s, %(A)s, "
                        "%(INCX)d, %(B)s, %(INCY)d)")

    def codemap(self, names, assumptions=True):
        varnames = 'alpha A B'.split()
        alpha, A, B = self.args

        namemap  = dict(zip(varnames, names))
        other = {'N': A.rows*A.cols,
                 'fn': self.fnname(self.typecode),
                 'INCX': 1,
                 'INCY': 1}
        return merge(namemap, other)

class SYRK(BLAS):
    """ Symmetric Rank-K Update `alpha X' X + beta Y' """
    def __init__(self, alpha, A, beta, D, typecode='D'):
        if isinstance(D, ZeroMatrix):
            D = ZeroMatrix(A.rows, A.rows)
        if isinstance(alpha, int):    alpha = float(alpha)
        if isinstance(beta, int):     beta = float(beta)

        return super(SYRK, self).__init__(alpha, A, beta, D, typecode)

    _inputs = (alpha, A, beta, D)
    _outputs = (alpha * A * A.T + beta * D,)
    inplace  = {0:3}
    condition = True

    @property
    def inputs(self):
        alpha, A, beta, D = self.args
        if isinstance(D, ZeroMatrix):    # special case this
            D = ZeroMatrix(A.rows, A.rows)
        # Sometimes we use C only as an output. It should be detransposed
        if isinstance(A, Transpose) and not isinstance(D, Transpose):
          A = detranspose(A)
        return alpha, A, beta, D

    @property
    def outputs(self):
        alpha, A, beta, D = self.args
        if isinstance(D, ZeroMatrix):    # special case this
            D = ZeroMatrix(A.rows, A.rows)
        return (alpha*A*A.T + beta*D,)

    fortran_template = ("call %(fn)s('%(UPLO)s', '%(TRANS)s', %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(beta)s, %(D)s, %(LDD)s)")

    def codemap(self, names, assumptions=True):
        varnames = 'alpha A beta D'.split()
        alpha, A, beta, D = self.args

        namemap  = dict(zip(varnames, names))
        other = {'TRANS': trans(A), 'LDA': LD(A), 'LDD': LD(D),
                 'N':str(A.shape[0]), 'K':str(A.shape[1]),
                 'fn': self.fnname(self.typecode),
                 'UPLO': 'U'} # TODO: symmetric matrices might be stored low
        return merge(namemap, other)

class COPY(BLAS, Copy):
    """ Array to array copy """
    _inputs   = (X,)
    _outputs  = (X,)

    fortran_template = "call %(fn)s(%(N)s, %(X)s, %(INCX)s, %(Y)s, %(INCY)s)"

    def codemap(self, names, assumptions=True):
        varnames = 'X Y'.split()
        X, = self.args

        namemap  = dict(zip(varnames, names))
        other = {'N': X.rows*X.cols,
                 'fn': self.fnname(self.typecode),
                 'INCX': 1,
                 'INCY': 1}
        return merge(namemap, other)

    @staticmethod
    def arguments(inputs, outputs):
        return inputs + outputs

    def fortran_call(self, input_names, output_names):
        return type(self).fortran_template % self.codemap(input_names+output_names)
