from computations import Computation
from computations.util import unique
from sympy import Symbol, Expr, Basic, ask, Tuple, S
from sympy.matrices.expressions import MatrixExpr, MatMul, ZeroMatrix
from sympy.strategies.tools import subs
from strategies import exhaust, chain
from sympy.strategies.traverse import bottom_up

def is_number(x):
    """ Is either a Python number or a SymPy number """
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

bases = [Expr, MatrixExpr]
def basetype(expr):
    for base in bases:
        if isinstance(expr, base):
            return base

def remove_numbers(coll):
    """ Remove numbers from a collection

    >>> from computations.matrices.core import remove_numbers
    >>> remove_numbers([1, 'x', 5, 'y'])
    ['x', 'y']
    """
    return filter(lambda x: not is_number(x), coll)

basetypes = {'S': 'real*4',
             'D': 'real*8',
             'C': 'complex*8',
             'Z': 'complex*16'}

def defloat(x):
    return S(x) if isinstance(x, (int, float)) else x

class MatrixCall(Computation):
    """ An atomic call, superclass for BLAS and LAPACK """
    def __init__(self, *args):
        if args[-1] not in basetypes:
            typecode = 'D'
        else:
            args, typecode = args[:-1], args[-1]
        self.args = tuple(map(defloat, args))
        self.typecode = typecode

    inputs = property(lambda self: self.args)

    @property
    def outputs(self):
        cls = self.__class__
        mapping = dict(zip(cls._inputs, self.inputs))
        return tuple([canonicalize(o.xreplace(mapping)) for o in cls._outputs])

    basetype = property(lambda self:  basetypes[self.typecode])
    _in_types = property(lambda self: (None,)*len(self._inputs))
    _out_types = property(lambda self: (None,)*len(self._outputs))

    in_types  = property(lambda self:
                          tuple(it or self.basetype for it in self._in_types))
    out_types = property(lambda self:
                          tuple(ot or self.basetype for ot in self._out_types))

    @classmethod
    def valid(cls, inputs, assumptions=True):
        d = dict(zip(cls._inputs, inputs))
        if cls.condition is True:
            return True
        return ask(cls.condition.xreplace(d), assumptions)

    @classmethod
    def fnname(cls, typecode):
        return typecode+cls.__name__.lower()

    @staticmethod
    def arguments(inputs, outputs):
        return inputs

    def fortran_call(self, input_names, output_names):
        args = type(self).arguments(self.inputs, self.outputs)
        name_map = dict(zip(self.inputs+self.outputs, input_names+output_names))
        argnames = [print_number(a) if is_number(a) else name_map[a] for a in args]
        codemap = self.codemap(argnames)
        return [self.fortran_template % codemap]

    def cuda_call(self, input_names, output_names):
        args = type(self).arguments(self.inputs, self.outputs)
        name_map = dict(zip(self.inputs+self.outputs, input_names+output_names))
        argnames = [print_number(a) if is_number(a) else name_map[a]+"_gpu" for a in args]
        codemap = self.codemap(argnames)
        return [self.cuda_template % codemap]


    def typecheck(self):
        return all(basetype(i) == basetype(_i)
                for (i, _i) in zip(self.inputs, self._inputs))


def fortran_double_str(x):
    if 'e' in str(x):
        return str(x).replace('e', 'd')
    else:
        return str(x) + 'd+0'


def print_number(x):
    if isinstance(x, float) or isinstance(x, Basic) and x.is_Float:
        return fortran_double_str(x)
    return str(x)


def nameof(var):
    """ Fortran name of variable """
    if is_number(var.expr):
        return var.expr
    else:
        return var.token

def canonicalize(x):
    if isinstance(x, MatrixExpr):
        return x.doit()
    if isinstance(x, Symbol):
        return x
    if isinstance(x, Expr):
        try:
            return type(x)(*x.args)
        except:
            return x
    else:
        return x
