from sympy import Expr, ZeroMatrix

def is_number(x):
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

def constant_arg(arg):
    """ Is this argument a constant?

    If so we don't want to include it as a parameter """
    return (is_number(arg) or isinstance(arg, ZeroMatrix) or 'MPI_' in
            str(arg))


def update_class(old, new):
    for k, v in new.__dict__.items():
        if '__' not in k:
            setattr(old, k, v)

def join(L):
    return '  ' + '\n  '.join([x for x in L if x])

