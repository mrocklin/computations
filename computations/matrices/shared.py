from sympy.matrices.expressions import Transpose
from sympy import ask, Q

def detranspose(A):
    """ Unpack a transposed matrix """
    if isinstance(A, Transpose):
        return A.arg
    else:
        return A

def trans(A):
    """ Return 'T' if A is a transpose, else 'N' """
    if isinstance(A, Transpose):
        return 'T'
    else:
        return 'N'

def uplo(A, assumptions):
    """ Return 'U' if A is stored in the upper Triangular 'U' if lower """
    if ask(Q.upper_triangular(A), assumptions):
        return 'U'
    if ask(Q.lower_triangular(A), assumptions):
        return 'L'

def LD(A):
    """ Leading dimension of matrix """
    # TODO make sure we don't use transposed matrices in untransposable slots
    return str(detranspose(A).shape[0])

def left_or_right(A, B, predicate, assumptions):
    """ Return 'L' if predicate is true of A, 'R' if predicate is true of B """
    if ask(predicate(A), assumptions):
        return 'L'
    if ask(predicate(B), assumptions):
        return 'R'

def diag(A, assumptions):
    """ Return 'U' if A is unit_triangular """
    if ask(Q.unit_triangular(A), assumptions):
        return 'U'
    else:
        return 'N'
