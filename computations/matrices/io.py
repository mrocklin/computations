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
