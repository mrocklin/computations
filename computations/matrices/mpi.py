from sympy import Symbol
from sympy.matrices import MatrixSymbol
from computations.matrices.core import MatrixCall
from computations import Computation

n, m = map(Symbol, 'nm')
A = MatrixSymbol('A', n, m)

class MatrixIntegerSymbol(MatrixSymbol):
    def fortran_type(self):
        return 'integer'

def new_status():
    new_status.i += 1
    return MatrixIntegerSymbol('status_%d' % new_status.i)
new_status.i = 0

class IntegerSymbol(Symbol):
    def fortran_type(self):
        return 'integer'

def new_ierr():
    new_ierr.i += 1
    return IntegerSymbol('ierr_%d' % new_ierr.i)
new_ierr.i = 0
def new_tag():
    new_tag.i += 1
    return IntegerSymbol('tag_%d' % new_tag.i)
new_tag.i = 0
def new_status():
    new_status.i += 1
    return MatrixIntegerSymbol('status_%d' % new_status.i,
                               Symbol('MPI_STATUS_SIZE', integer=True), 1)
new_status.i = 0


class Send(Computation):
    def __init__(self, data, dest, tag=None, ierr=None):
        ierr = ierr or new_ierr()
        self.tag = tag or new_tag()
        self.dest = dest

        self.inputs = (data,)
        self.outputs = (ierr,)

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s-->%s"]' % (
                str(self), str(self.__class__.__name__), str(self.dest))


class Recv(Computation):
    def __init__(self, data, source, tag=None, status=None, ierr=None):
        ierr = ierr or new_ierr()
        status = status or new_status()
        self.tag = tag or new_tag()
        self.source = source

        self.inputs = ()
        self.outputs = (data, ierr, status)

    def _write_dot(self):
        return '"%s" [shape=diamond, label="%s<--%s"]' % (
                str(self), str(self.__class__.__name__), str(self.source))


tagdb = dict()
def gettag(a, b, expr):
    """ MPI Tag associated to transfer of expr from machine a to machine b """
    if (a, b, expr) not in tagdb:
        tagdb[(a, b, expr)] = gettag._tag
        gettag._tag += 1
    return tagdb[(a, b, expr)]
gettag._tag = 1


def send(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    for v in sharedvars:
        # TODO: deal with multiple variables
        tag = gettag(from_machine, to_machine, v)
        return Send(v, to_machine, tag=tag)

def recv(from_machine, to_machine, from_job, to_job):
    sharedvars = set(from_job.outputs).intersection(set(to_job.inputs))
    for v in sharedvars:
        # TODO: deal with multiple variables
        tag = gettag(from_machine, to_machine, v)
        return Recv(v, from_machine, tag=tag)
