from sympy import Symbol
from sympy.matrices import MatrixSymbol
from computations.matrices.core import MatrixCall
from computations import Computation

n, m = map(Symbol, 'nm')
A = MatrixSymbol('A', n, m)

def new_fid():
    new_fid.i += 1
    return Symbol('fid_%d' % new_fid.i, integer=True)
new_fid.i = 0

class ReadFromFile(Computation):
    def __init__(self, filename, output, identifier=None):
        identifier = identifier or new_fid()
        self.inputs = ()
        self.outputs = (output, identifier)
        self.filename = filename

    def fortran_call(self, input_names, output_names):
        output, fid = output_names
        filename = self.filename
        d = locals()
        return ['open(newunit=%(fid)s, file="%(filename)s", status="old")'%d,
                'read(%(fid)s, *) %(output)s'%d,
                'close(%(fid)s)'%d]

    def pseudocode_call(self, input_names, output_names):
        output, fid = output_names
        filename = self.filename
        return ['%(output)s := Read from %(filename)s' % locals()]

    def typecheck(self):
        return isinstance(self.inputs, MatrixExpr)


class WriteToFile(Computation):
    def __init__(self, filename, input, identifier=None):
        identifier = identifier or new_fid()
        self.inputs = (input,)
        self.outputs = (identifier,)
        self.filename = filename

    def fortran_call(self, input_names, output_names):
        input, = input_names
        fid, = output_names
        filename = self.filename
        d = locals()
        return ['open(newunit=%(fid)s, file="%(filename)s", status="replace")'%d,
                'write(%(fid)s, *) %(input)s'%d,
                'close(%(fid)s)'%d]

    def pseudocode_call(self, input_names, output_names):
        input, = input_names
        filename = self.filename
        return ['Write %(input)s to %(filename)s' % locals()]

    def typecheck(self):
        return isinstance(self.inputs, MatrixExpr)


def disk_io(comp, filenames):
    """ Return a computation with reads/writes covering inputs/outputs

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> from computations.matrices.blas import GEMM
    >>> from computations.matrices.io import disk_io
    >>> A = MatrixSymbol('A', 3, 4)
    >>> B = MatrixSymbol('B', 4, 5)
    >>> gemm = GEMM(1, A, B, 0, ZeroMatrix(3, 5))
    >>> disk_io(gemm, {A: 'A.dat', B: 'B.dat', A*B: 'AB.dat'}) # doctest: +SKIP
    [[[] -> ReadFromFile -> [A, fid_2]
      [] -> ReadFromFile -> [B, fid_1]
      [1.00000000000000, A, B, 0.0, 0] -> GEMM -> [A*B]
      [A*B] -> WriteToFile -> [fid_3]]]
    """
    d = filenames
    reads  = [ReadFromFile(d[i], i) for i in d if i in comp.inputs]
    writes = [WriteToFile(d[o], o) for o in d if o in comp.outputs]
    return sum(reads + writes, comp)
